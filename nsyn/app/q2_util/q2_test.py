import unittest

from nsyn.app.q2_util.q2_grammar import Query2


class TestQuery2Parsing(unittest.TestCase):
    def test_parse_query_sample_1(self) -> None:
        query = """
        SELECT mean(M1) FROM adult WHERE M2 > 0 AND adult.edu == "bachelor" GROUP BY M3 ! 0.5
        M1: adult-model1, autogluon
        M2: adult-model2, autogluon
        M3: adult-model3, llm
        """
        result = Query2.parse_query(query)
        self.assertEqual(result.dataset_name, "adult")
        self.assertEqual(result.projection_model, ("adult-model1", "autogluon"))
        self.assertEqual(result.projection_model_agg_func, "mean")
        self.assertEqual(result.selection_models, [("adult-model2", "autogluon")])
        self.assertEqual(result.selection_model_conditions, ["> 0"])
        self.assertEqual(result.group_by_model, ("adult-model3", "llm"))
        self.assertEqual(result.group_by_partition_threshold, 0.5)
        self.assertIsNone(result.legacy_projection_column)
        self.assertIsNone(result.legacy_projection_agg_func)
        self.assertEqual(result.legacy_selection_filters, [("edu", '== "bachelor"')])
        self.assertIsNone(result.legacy_group_by_column)

    def test_parse_query_sample_2(self) -> None:
        query = """
        SELECT adult.* FROM adult WHERE M1 == "low" AND adult.edu == "bachelor" GROUP BY adult.child_num
        M1: adult-model1, autogluon
        """
        result = Query2.parse_query(query)
        self.assertEqual(result.dataset_name, "adult")
        self.assertIsNone(result.projection_model)
        self.assertIsNone(result.projection_model_agg_func)
        self.assertEqual(result.selection_models, [("adult-model1", "autogluon")])
        self.assertEqual(result.selection_model_conditions, ['== "low"'])
        self.assertIsNone(result.group_by_model)
        self.assertIsNone(result.group_by_partition_threshold)
        self.assertEqual(result.legacy_projection_column, "*")
        self.assertIsNone(result.legacy_projection_agg_func)
        self.assertEqual(result.legacy_selection_filters, [("edu", '== "bachelor"')])
        self.assertEqual(result.legacy_group_by_column, "child_num")

    def test_parse_query_sample_3(self) -> None:
        query = """
        SELECT median(adult.age) FROM adult WHERE adult.edu == "bachelor" GROUP BY M1
        M1: adult-model1, llm
        """
        result = Query2.parse_query(query)
        self.assertEqual(result.dataset_name, "adult")
        self.assertIsNone(result.projection_model)
        self.assertIsNone(result.projection_model_agg_func)
        self.assertIsNone(result.selection_models)
        self.assertIsNone(result.selection_model_conditions)
        self.assertEqual(result.group_by_model, ("adult-model1", "llm"))
        self.assertIsNone(result.group_by_partition_threshold)
        self.assertEqual(result.legacy_projection_column, "age")
        self.assertEqual(result.legacy_projection_agg_func, "median")
        self.assertEqual(result.legacy_selection_filters, [("edu", '== "bachelor"')])
        self.assertIsNone(result.legacy_group_by_column)


if __name__ == "__main__":
    unittest.main()
