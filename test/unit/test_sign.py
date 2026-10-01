# Copyright 2022 The Sigstore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import secrets

import cryptography.x509 as x509
import pretend
import pytest
import rfc3161_client
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from sigstore_models.common.v1 import HashAlgorithm

import sigstore.oidc
from sigstore._internal.timestamp import TimestampError
from sigstore.dsse import StatementBuilder, Subject
from sigstore.errors import Error, VerificationError
from sigstore.hashes import Hashed
from sigstore.models import Bundle, SigningConfig
from sigstore.sign import Signer, SigningContext
from sigstore.verify.policy import UnsafeNoOp


def _signer_with_identity(identity: str) -> Signer:
    """
    Build a `Signer` wired up just enough to exercise CSR construction,
    without any network access or a real identity token.
    """
    signer = Signer.__new__(Signer)
    signer._identity_token = pretend.stub(_identity=identity)
    signer._Signer__cached_private_key = ec.generate_private_key(ec.SECP256R1())
    return signer


def test_build_csr_has_empty_subject():
    # The CSR carries an empty subject regardless of the identity. Fulcio does
    # not use the subject (sigstore/fulcio#863), so we omit it entirely.
    csr = _signer_with_identity("foo@example.com")._build_csr()

    assert len(csr.subject) == 0

    der = csr.public_bytes(serialization.Encoding.DER)
    reparsed = x509.load_der_x509_csr(der)
    assert len(reparsed.subject) == 0


def test_build_csr_non_ascii_identity_produces_valid_csr():
    # Regression test for sigstore/sigstore-python#1507. A non-ASCII identity
    # (e.g. a GitHub Actions `sub` with a non-ASCII environment name) must not
    # be smuggled into an ASCII-only IA5String EMAIL_ADDRESS attribute, which
    # produces malformed DER that Fulcio rejects with HTTP 400. With the subject
    # omitted, the identity never reaches the CSR.
    identity = "repo:foo/bar:environment:prod-\U0001f600"
    assert any(ord(ch) > 0x7F for ch in identity)

    csr = _signer_with_identity(identity)._build_csr()

    assert len(csr.subject) == 0

    # The CSR serializes to DER and re-parses cleanly, and no subject attribute
    # carries non-ASCII bytes (the malformed-DER condition Fulcio rejects).
    der = csr.public_bytes(serialization.Encoding.DER)
    reparsed = x509.load_der_x509_csr(der)
    assert len(reparsed.subject) == 0


@pytest.mark.staging
@pytest.mark.ambient_oidc
def test_signer_uses_valid_cached_certificate_with_expired_identity(staging):
    ctx_cls, _, identity = staging
    ctx: SigningContext = ctx_cls()
    assert identity is not None

    # Constructing a cached signer obtains its certificate while the token is
    # valid. Expire the local token afterward, then perform a real signing
    # operation: it must reuse the still-valid cached certificate.
    with ctx.signer(identity) as signer:
        identity._exp = 0
        assert not signer._identity_token.in_validity_period()
        signer.sign_artifact(secrets.token_bytes(32))


# only check the log contents for production: staging is already on
# rekor v2 and we don't currently support log lookups on rekor v2.
# This test can likely be removed once prod also uses rekor v2
@pytest.mark.parametrize("env", ["production"])
@pytest.mark.ambient_oidc
def test_sign_rekor_entry_consistent(request, sign_ctx_and_ident_for_env):
    ctx_cls, identity = sign_ctx_and_ident_for_env

    # NOTE: The actual signer instance is produced lazily, so that parameter
    # expansion doesn't fail in offline tests.
    ctx: SigningContext = ctx_cls()
    assert identity is not None

    payload = secrets.token_bytes(32)
    with ctx.signer(identity) as signer:
        expected_entry = signer.sign_artifact(payload).log_entry

    actual_entry = ctx._rekor.log.entries.get(log_index=expected_entry._inner.log_index)

    assert (
        expected_entry._inner.canonicalized_body
        == actual_entry._inner.canonicalized_body
    )
    assert expected_entry._inner.integrated_time == actual_entry._inner.integrated_time
    assert expected_entry._inner.log_id == actual_entry._inner.log_id
    assert expected_entry._inner.log_index == actual_entry._inner.log_index


@pytest.mark.staging
@pytest.mark.ambient_oidc
def test_sign_with_staging(staging, null_policy):
    ctx_cls, verifier_cls, identity = staging

    ctx: SigningContext = ctx_cls()
    verifier = verifier_cls()
    assert identity is not None

    payload = secrets.token_bytes(32)
    with ctx.signer(identity) as signer:
        bundle = signer.sign_artifact(payload)

    verifier.verify_artifact(payload, bundle, null_policy)


@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.ambient_oidc
def test_sct_verify_keyring_lookup_error(sign_ctx_and_ident_for_env, monkeypatch):
    ctx, identity = sign_ctx_and_ident_for_env

    # a signer whose keyring always fails to lookup a given key.
    ctx: SigningContext = ctx()
    mock = pretend.stub(
        ct_keyring=lambda *a: pretend.stub(verify=pretend.raiser(VerificationError))
    )
    ctx._trusted_root = mock
    assert identity is not None

    payload = secrets.token_bytes(32)
    with pytest.raises(VerificationError, match=r"SCT verify failed:"):
        with ctx.signer(identity) as signer:
            signer.sign_artifact(payload)


@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.ambient_oidc
def test_sct_verify_keyring_error(sign_ctx_and_ident_for_env, monkeypatch):
    ctx, identity = sign_ctx_and_ident_for_env

    # a signer whose keyring throws an internal error.
    ctx: SigningContext = ctx()
    mock = pretend.stub(
        ct_keyring=lambda *a: pretend.stub(verify=pretend.raiser(VerificationError))
    )
    ctx._trusted_root = mock
    assert identity is not None

    payload = secrets.token_bytes(32)

    with pytest.raises(VerificationError):
        with ctx.signer(identity) as signer:
            signer.sign_artifact(payload)


@pytest.mark.parametrize("env", ["staging", "production"])
@pytest.mark.ambient_oidc
def test_identity_proof_fallback_claim(sign_ctx_and_ident_for_env, monkeypatch):
    ctx_cls, identity = sign_ctx_and_ident_for_env

    ctx: SigningContext = ctx_cls()
    assert identity is not None

    # clear out known issuers, forcing the `Identity`'s  `sub` claim to be used
    # as fall back
    monkeypatch.setattr(sigstore.oidc, "_KNOWN_OIDC_ISSUERS", {})

    payload = secrets.token_bytes(32)

    with ctx.signer(identity) as signer:
        signer.sign_artifact(payload)


@pytest.mark.staging
@pytest.mark.ambient_oidc
def test_sign_prehashed(staging):
    sign_ctx_cls, verifier_cls, identity = staging

    sign_ctx = sign_ctx_cls()
    verifier = verifier_cls()

    input_ = secrets.token_bytes(32)
    hashed = Hashed(
        digest=hashlib.sha256(input_).digest(), algorithm=HashAlgorithm.SHA2_256
    )

    with sign_ctx.signer(identity) as signer:
        bundle = signer.sign_artifact(hashed)

    assert bundle._inner.message_signature.message_digest.algorithm == hashed.algorithm
    assert bundle._inner.message_signature.message_digest.digest == hashed.digest

    # verifying against the original input works
    verifier.verify_artifact(input_, bundle=bundle, policy=UnsafeNoOp())
    # verifying against the prehash also works
    verifier.verify_artifact(hashed, bundle=bundle, policy=UnsafeNoOp())


@pytest.mark.staging
@pytest.mark.ambient_oidc
def test_sign_dsse(staging):
    sign_ctx, _, identity = staging

    ctx = sign_ctx()
    stmt = (
        StatementBuilder()
        .subjects(
            [Subject(name="null", digest={"sha256": hashlib.sha256(b"").hexdigest()})]
        )
        .predicate_type("https://cosign.sigstore.dev/attestation/v1")
        .predicate(
            {
                "Data": "",
                "Timestamp": "2023-12-07T00:37:58Z",
            }
        )
    ).build()

    with ctx.signer(identity) as signer:
        bundle = signer.sign_dsse(stmt)
        # Ensures that all of our inner types serialize as expected.
        bundle.to_json()


class TestSignWithoutTransparencyLog:
    """
    Signing against a Sigstore instance that has a timestamp authority but no
    transparency log (e.g. GitHub's instance for private repositories).

    Fulcio, the TSA and Rekor are stubbed, so these run offline.
    """

    @pytest.fixture
    def tsa_bundle(self, asset) -> Bundle:
        return Bundle.from_json(asset("tsa/bundle.txt.sigstore").read_bytes())

    @pytest.fixture
    def timestamp_response(self, tsa_bundle):
        tvd = tsa_bundle.verification_material.timestamp_verification_data
        return tvd._inner.rfc3161_timestamps[0].signed_timestamp

    @staticmethod
    def _tsa(response=None):
        def request_timestamp(signature):
            if response is None:
                raise TimestampError("TSA unavailable")
            return rfc3161_client.decode_timestamp_response(response)

        return pretend.stub(
            url="https://tsa.example.com", request_timestamp=request_timestamp
        )

    @staticmethod
    def _signer(ctx: SigningContext, cert: x509.Certificate) -> Signer:
        signer = Signer.__new__(Signer)
        signer._signing_ctx = ctx
        signer._Signer__cached_private_key = ec.generate_private_key(ec.SECP256R1())
        # The fixture certificate is long expired; sidestep the expiry check.
        signer._signing_cert = lambda: cert
        return signer

    @staticmethod
    def _ctx(rekor, tsa_clients) -> SigningContext:
        return SigningContext(
            fulcio=pretend.stub(),
            rekor=rekor,
            trusted_root=pretend.stub(),
            tsa_clients=tsa_clients,
        )

    @staticmethod
    def _statement():
        return (
            StatementBuilder()
            .subjects(
                [
                    Subject(
                        name="null", digest={"sha256": hashlib.sha256(b"").hexdigest()}
                    )
                ]
            )
            .predicate_type("https://cosign.sigstore.dev/attestation/v1")
            .predicate({})
        ).build()

    def test_context_requires_tlog_or_tsa(self):
        with pytest.raises(Error, match="needs a transparency log, a timestamp"):
            self._ctx(rekor=None, tsa_clients=[])

    def test_context_from_trust_config_without_tlog(self, asset):
        trust_config = pretend.stub(
            signing_config=SigningConfig.from_file(
                asset("signing_config/signingconfig-no-tlog.v2.json")
            ),
            trusted_root=pretend.stub(),
        )

        ctx = SigningContext.from_trust_config(trust_config)

        assert ctx._rekor is None
        assert len(ctx._tsa_clients) == 1

    @pytest.mark.parametrize("sign", ["artifact", "dsse"])
    def test_sign_without_tlog(self, sign, tsa_bundle, timestamp_response):
        ctx = self._ctx(rekor=None, tsa_clients=[self._tsa(timestamp_response)])
        signer = self._signer(ctx, tsa_bundle.signing_certificate)

        if sign == "artifact":
            bundle = signer.sign_artifact(b"hello")
        else:
            bundle = signer.sign_dsse(self._statement())

        assert not bundle.has_log_entry
        assert bundle._inner.verification_material.tlog_entries == []
        tvd = bundle.verification_material.timestamp_verification_data
        assert tvd is not None
        assert len(tvd.rfc3161_timestamps) == 1

        # The bundle survives serialization, which re-validates its layout.
        assert not Bundle.from_json(bundle.to_json()).has_log_entry

    def test_sign_without_tlog_tolerates_a_failing_tsa(
        self, tsa_bundle, timestamp_response
    ):
        ctx = self._ctx(
            rekor=None, tsa_clients=[self._tsa(), self._tsa(timestamp_response)]
        )
        signer = self._signer(ctx, tsa_bundle.signing_certificate)

        bundle = signer.sign_artifact(b"hello")

        tvd = bundle.verification_material.timestamp_verification_data
        assert tvd is not None
        assert len(tvd.rfc3161_timestamps) == 1

    def test_sign_without_tlog_requires_a_timestamp(self, tsa_bundle):
        ctx = self._ctx(rekor=None, tsa_clients=[self._tsa()])
        signer = self._signer(ctx, tsa_bundle.signing_certificate)

        with pytest.raises(Error, match="No timestamp authority returned"):
            signer.sign_artifact(b"hello")

    def test_sign_with_tlog_still_submits_entry(self, tsa_bundle, timestamp_response):
        # With a transparency log configured, the entry is built against that
        # log and submitted, exactly as before.
        proposed = object()
        rekor = pretend.stub(
            _build_hashed_rekord_request=pretend.call_recorder(
                lambda **kwargs: proposed
            ),
            create_entry=pretend.call_recorder(lambda entry: tsa_bundle.log_entry),
        )
        ctx = self._ctx(rekor=rekor, tsa_clients=[self._tsa(timestamp_response)])
        signer = self._signer(ctx, tsa_bundle.signing_certificate)

        bundle = signer.sign_artifact(b"hello")

        assert len(rekor._build_hashed_rekord_request.calls) == 1
        assert rekor.create_entry.calls == [pretend.call(proposed)]
        assert bundle.has_log_entry
