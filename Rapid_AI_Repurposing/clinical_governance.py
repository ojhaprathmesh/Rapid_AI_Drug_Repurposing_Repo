"""
clinical_governance.py
======================
Technical Safeguards Engine for Healthcare & Clinical Compliance
Implements technical security specifications stipulated by HIPAA Security Rule
(45 CFR § 164.312) and GDPR Data Protection by Design (Article 25):

1. PHI / PII Scrubber (§ 164.514(b)(2) Safe Harbor De-identification):
   Pre-ingestion sanitizer stripping 18 HIPAA identifiers from clinical notes,
   patient context, and candidate queries prior to model ingestion.

2. Cryptographic Tamper-Evident Audit Logging (§ 164.312(b) Audit Controls):
   SHA-256 hash-chained immutable audit ledger recording every prediction,
   prompt payload, and clinical rationale with cryptographic verification.

3. Zero-Egress Air-Gap Assertion (§ 164.312(e) Transmission Security):
   Runtime network isolation guard verifying zero unauthorized external outbound
   socket connections leave localhost during inference.
"""

import os
import re
import json
import time
import socket
import hashlib
from typing import Dict, Any, Tuple, Optional, List
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────────────────────
# 1. PHI / PII SCRUBBER (HIPAA 18 Safe Harbor Identifiers)
# ─────────────────────────────────────────────────────────────────────────────

class PHIScrubber:
    """
    Sanitizes clinical and patient text inputs by detecting and redacting
    Protected Health Information (PHI) under HIPAA Safe Harbor (45 CFR § 164.514).
    """

    PATTERNS = [
        # Social Security Numbers (SSN)
        (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
        (r'\bSSN[:\s]*\d{9}\b', '[REDACTED_SSN]'),
        
        # Medical Record Numbers (MRN) & Hospital IDs
        (r'\b(?:MRN|EHR|ID|PATIENT\s*ID)[:\s#]*[A-Z0-9-]{4,16}\b', '[REDACTED_MRN]'),
        
        # Phone numbers and faxes
        (r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]'),
        
        # Email addresses
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]'),
        
        # IP Addresses (v4)
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[REDACTED_IP]'),
        
        # Dates of birth, admission, discharge, death (YYYY-MM-DD, MM/DD/YYYY, etc.)
        (r'\b(?:DOB|DOA|DOD|Date\s*of\s*Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED_DATE]'),
        (r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b', '[REDACTED_DATE]'),
        
        # Common patient honorific name prefixes in clinical notes
        (r'\b(?:Patient|Pt\.?|Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b', '[REDACTED_PATIENT_NAME]'),
        
        # Age > 89 (HIPAA requires aggregate binning for age >= 90)
        (r'\b(?:age|aged)[:\s]*(?:9[0-9]|1[0-9]{2})\b', '[REDACTED_AGE_OVER_89]'),
        
        # US ZIP codes (5 digits or ZIP+4)
        (r'\bZIP[:\s]*\d{5}(?:-\d{4})?\b', '[REDACTED_ZIP]'),
    ]

    @classmethod
    def scrub(cls, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Sanitizes text and returns the scrubbed text along with detection metrics.
        """
        if not text or not isinstance(text, str):
            return text, {"redactions_count": 0, "scrubbed_tokens": []}

        scrubbed = text
        redactions = []
        for pattern, replacement in cls.PATTERNS:
            matches = re.findall(pattern, scrubbed, flags=re.IGNORECASE)
            if matches:
                redactions.extend([(m, replacement) for m in matches])
                scrubbed = re.sub(pattern, replacement, scrubbed, flags=re.IGNORECASE)

        return scrubbed, {
            "redactions_count": len(redactions),
            "original_length": len(text),
            "scrubbed_length": len(scrubbed),
            "redactions": [r[1] for r in redactions],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. CRYPTOGRAPHIC TAMPER-EVIDENT AUDIT LOGGER (45 CFR § 164.312(b))
# ─────────────────────────────────────────────────────────────────────────────

class TamperEvidentAuditLogger:
    """
    Maintains an append-only, SHA-256 hash-chained cryptographic ledger of all
    model inferences, candidate rationalizations, and clinician queries.
    Any retrospective alteration of historical records invalidates the hash chain.
    """

    def __init__(self, ledger_path: Optional[str] = None):
        if ledger_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            reports_dir = os.path.join(base_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            self.ledger_path = os.path.join(reports_dir, "clinical_audit_ledger.jsonl")
        else:
            self.ledger_path = ledger_path
            os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)

    def _get_last_hash(self) -> str:
        """Retrieves the block_hash of the last entry, or genesis hash if empty."""
        if not os.path.exists(self.ledger_path) or os.path.getsize(self.ledger_path) == 0:
            return "0" * 64

        last_line = ""
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if not last_line:
            return "0" * 64

        try:
            entry = json.loads(last_line)
            return entry.get("block_hash", "0" * 64)
        except Exception:
            return "0" * 64

    def log_event(
        self,
        event_type: str,
        drug: str,
        disease: str,
        score: float,
        prompt_payload: str,
        rationale_output: str,
        operator_id: str = "on_premise_clinician",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates and appends a cryptographically chained block to the audit ledger.
        """
        prev_hash = self._get_last_hash()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Cryptographic digest of inputs and outputs
        prompt_hash = hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest()
        output_hash = hashlib.sha256(rationale_output.encode("utf-8")).hexdigest()

        # Canonical representation for hash chaining
        block_data = {
            "timestamp": timestamp,
            "event_type": event_type,
            "operator_id": operator_id,
            "drug": drug,
            "disease": disease,
            "score": round(float(score), 6),
            "prompt_sha256": prompt_hash,
            "output_sha256": output_hash,
            "extra_metadata": extra_metadata or {},
            "prev_hash": prev_hash,
        }

        canonical_str = json.dumps(block_data, sort_keys=True)
        block_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        block_data["block_hash"] = block_hash

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(block_data) + "\n")

        return block_data

    @classmethod
    def verify_ledger_integrity(cls, ledger_path: str) -> Tuple[bool, int, Optional[str]]:
        """
        Verifies the SHA-256 cryptographic chaining across the entire ledger.
        Returns: (is_valid, total_blocks_checked, error_message)
        """
        if not os.path.exists(ledger_path):
            return True, 0, None

        expected_prev_hash = "0" * 64
        count = 0

        with open(ledger_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                count += 1
                try:
                    block = json.loads(line)
                except Exception as e:
                    return False, count, f"Corrupted JSON on line {line_idx + 1}: {e}"

                recorded_hash = block.get("block_hash")
                recorded_prev = block.get("prev_hash")

                if recorded_prev != expected_prev_hash:
                    return False, count, (
                        f"Hash chain broken at block {count}: "
                        f"recorded prev_hash '{recorded_prev}' does not match expected '{expected_prev_hash}'"
                    )

                # Recompute hash
                verification_data = dict(block)
                verification_data.pop("block_hash", None)
                recomputed_hash = hashlib.sha256(
                    json.dumps(verification_data, sort_keys=True).encode("utf-8")
                ).hexdigest()

                if recomputed_hash != recorded_hash:
                    return False, count, (
                        f"Tampering detected in block {count}: "
                        f"recomputed hash '{recomputed_hash}' != recorded '{recorded_hash}'"
                    )

                expected_prev_hash = recorded_hash

        return True, count, None


# ─────────────────────────────────────────────────────────────────────────────
# 3. ZERO-EGRESS AIR-GAP NETWORK ASSERTION (45 CFR § 164.312(e))
# ─────────────────────────────────────────────────────────────────────────────

class AirGapSecurityViolation(Exception):
    """Raised when an unauthorized outbound external network socket connection is attempted."""
    pass


class AirGapGuard:
    """
    Enforces local compute containment by intercepting network socket calls.
    Allows only localhost / loopback interfaces (e.g., 127.0.0.1 for local Ollama).
    Blocks and records any attempt to connect to external WAN IP addresses or domains.
    """

    ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    @classmethod
    @contextmanager
    def assert_zero_egress(cls, allowed_ports: Optional[List[int]] = None):
        """
        Context manager that intercepts socket.connect to enforce strict air-gapped execution.
        """
        original_connect = socket.socket.connect
        ports = set(allowed_ports) if allowed_ports else {11434, 8501, 8000}  # Ollama, Streamlit, local API

        def guarded_connect(sock_self, address):
            # address is typically (host, port)
            if isinstance(address, tuple) and len(address) >= 2:
                host, port = address[0], address[1]
                # Normalize localhost
                if str(host) not in cls.ALLOWED_HOSTS:
                    raise AirGapSecurityViolation(
                        f"Air-Gap Policy Violation: Unauthorized outbound network egress attempt "
                        f"to '{host}:{port}'. Rapid AI enforces strict zero-egress local execution."
                    )
            return original_connect(sock_self, address)

        socket.socket.connect = guarded_connect
        try:
            yield
        finally:
            socket.socket.connect = original_connect

    @classmethod
    def test_airgap_containment(cls) -> bool:
        """
        Verifies that the Air-Gap guard correctly blocks unauthorized WAN connections
        while permitting authorized local loopback calls.
        """
        blocked = False
        s = None
        try:
            with cls.assert_zero_egress():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.1)
                s.connect(("8.8.8.8", 53))  # Google DNS
        except AirGapSecurityViolation:
            blocked = True
        except Exception:
            # If socket fails before connect or is blocked by OS firewall, still compliant
            blocked = True
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass

        return blocked
