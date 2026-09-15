import unittest
from xml.etree import ElementTree as ET

from module.strength import build_summary, make_row, candidate_records, descendant_values


class ExtractStrengthTests(unittest.TestCase):
    def test_namespace_and_strength_fields(self):
        root = ET.fromstring(
            '''<b:Boring xmlns:b="urn:test">
                <b:BoringId>B-001</b:BoringId>
                <b:X>100.5</b:X><b:Y>200.5</b:Y><b:孔口標高>12.0</b:孔口標高>
                <b:Sample>
                    <b:試料番号>S-01</b:試料番号>
                    <b:開始深度>2</b:開始深度><b:終了深度>4</b:終了深度>
                    <b:土質名>砂質土</b:土質名>
                    <b:StrengthTest><b:試験名>UU</b:試験名><b:粘着力全応力>20</b:粘着力全応力></b:StrengthTest>
                </b:Sample>
            </b:Boring>'''
        )
        records = candidate_records(root)
        self.assertEqual(len(records), 1)
        row = make_row(descendant_values(root), records[0], "one.xml")
        self.assertEqual(row["boring_id"], "B-001")
        self.assertEqual(row["depth_mid"], 3.0)
        self.assertEqual(row["test_type"], "UU")
        self.assertEqual(row["c_total"], 20)

    def test_summary_counts_present_values(self):
        rows = [{"boring_id": "B-1", "test_type": "CU", "c_total": 10, "phi_total": None,
                 "c_effective": None, "phi_effective": 30, "qu": None}]
        summary = build_summary(1, 1, 0, rows)
        self.assertEqual(summary["strength_records"], 1)
        self.assertEqual(summary["test_type_CU"], 1)
        self.assertEqual(summary["present_c_total"], 1)
        self.assertEqual(summary["present_phi_total"], 0)


if __name__ == "__main__":
    unittest.main()
