#!/usr/bin/env python3
"""
test_clinical_governance.py
===========================
Automated test suite for HIPAA 45 CFR § 164.312 Technical Safeguards:
- PHI Safe Harbor Redaction (§ 164.514(b))
- Cryptographic SHA-256 Audit Trail Integrity (§ 164.312(b))
- Zero-Egress Air-Gap Network Isolation (§ 164.312(e))
"""

import os
import sys
import tempfile
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from clinical_governance import PHIScrubber, TamperEvidentAuditLogger, AirGapGuard, AirGapSecurityViolation


class TestClinicalGovernance(unittest.TestCase):

    def test_phi_sanitization(self):
        sample_clinical_note = (
            "Patient Robert Johnson (DOB: 1958-11-23, SSN: 987-65-4321, MRN: MRN-401928) "
            "presented with severe hypertension. Contact: robert.j@healthmail.com, Tel: 555-123-4567. "
            "Evaluated at clinic IP 192.168.1.100."
        )
        scrubbed, meta = PHIScrubber.scrub(sample_clinical_note)

        self.assertNotIn("Robert Johnson", scrubbed)
        self.assertNotIn("987-65-4321", scrubbed)
        self.assertNotIn("MRN-401928", scrubbed)
        self.assertNotIn("1958-11-23", scrubbed)
        self.assertNotIn("robert.j@healthmail.com", scrubbed)
        self.assertNotIn("555-123-4567", scrubbed)
        self.assertNotIn("192.168.1.100", scrubbed)
        self.assertGreaterEqual(meta["redactions_count"], 6)

    def test_cryptographic_audit_ledger_integrity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = os.path.join(tmpdir, "test_ledger.jsonl")
            logger = TamperEvidentAuditLogger(ledger_path=ledger_path)

            logger.log_event("PREDICTION", "Niacin", "ocular hypertension", 0.9999, "P1", "R1")
            logger.log_event("PREDICTION", "Vardenafil", "hypertension", 0.9998, "P2", "R2")
            logger.log_event("PREDICTION", "Doxorubicin", "T-cell leukemia", 0.9998, "P3", "R3")

            # Check valid chain
            is_valid, count, err = TamperEvidentAuditLogger.verify_ledger_integrity(ledger_path)
            self.assertTrue(is_valid, f"Ledger integrity failed: {err}")
            self.assertEqual(count, 3)

            # Test detection of tampering
            with open(ledger_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Tamper with block 2
            tampered_b2 = lines[1].replace("0.9998", "0.1234")
            with open(ledger_path, "w", encoding="utf-8") as f:
                f.writelines([lines[0], tampered_b2, lines[2]])

            is_valid_t, count_t, err_t = TamperEvidentAuditLogger.verify_ledger_integrity(ledger_path)
            self.assertFalse(is_valid_t)
            self.assertIn("Tampering detected", err_t)

    def test_zero_egress_airgap_containment(self):
        blocked = AirGapGuard.test_airgap_containment()
        self.assertTrue(blocked, "AirGapGuard must intercept external WAN connections.")


if __name__ == "__main__":
    unittest.main()
