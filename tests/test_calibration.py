import unittest

from trustpanel.calibration import calibrate, fit_linear, tilt
from trustpanel.personas import evidence_quotas
from trustpanel.validation import expected


class CalibrationTest(unittest.TestCase):
    def test_fit_linear(self):
        self.assertEqual(fit_linear([1, 2, 3], [2, 4, 6]), (0.0, 2.0))
        self.assertEqual(fit_linear([3, 3], [4, 5]), (4.5, 0.0))  # no signal: predict the mean

    def test_tilt_hits_target_and_keeps_prior_at_its_own_mean(self):
        prior = [0.1, 0.05, 0.1, 0.15, 0.6]
        self.assertAlmostEqual(expected(tilt(prior, 3.5)), 3.5, places=4)
        for a, b in zip(tilt(prior, expected(prior)), prior):
            self.assertAlmostEqual(a, b, places=4)

    def test_calibrate_corrects_a_harsh_panel(self):
        prior = [0.1, 0.05, 0.1, 0.15, 0.6]

        def rec(pred_mean, real_mean):
            return {"persona_panel": tilt(prior, pred_mean), "truth": tilt(prior, real_mean), "topic_prior": prior}

        cal = [rec(m - 1, m) for m in (3.5, 4.0, 4.5)]  # panel is one star too harsh
        test = [rec(2.2, 3.2)]
        fitted = calibrate(test, cal, "persona_panel")
        self.assertAlmostEqual(fitted["slope"], 1.0, places=3)
        self.assertAlmostEqual(expected(test[0]["persona_calibrated"]), 3.2, places=2)


class EvidenceQuotaTest(unittest.TestCase):
    def test_proportional_with_a_complaint(self):
        quotas = evidence_quotas({1: 5, 2: 3, 3: 4, 4: 8, 5: 80}, 12)
        self.assertEqual(sum(quotas.values()), 12)
        self.assertGreaterEqual(quotas[5], 9)
        self.assertGreaterEqual(quotas[1] + quotas[2], 1)

    def test_no_complaints_to_show(self):
        self.assertEqual(evidence_quotas({4: 1, 5: 3}, 4), {1: 0, 2: 0, 3: 0, 4: 1, 5: 3})


if __name__ == "__main__":
    unittest.main()
