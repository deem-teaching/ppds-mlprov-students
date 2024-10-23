"""
Tests whether the adult_easy test pipeline works
"""
import ast
import os

from mlprov.utils import get_project_root

ADULT_COMPLEX_PY = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_complex.py")
AMAZON_REVIEWS_PY = os.path.join(str(get_project_root()), "example_pipelines", "amazon_reviews", "amazon_reviews.py")
COMPAS_PY = os.path.join(str(get_project_root()), "example_pipelines", "compas", "compas.py")
HEALTHCARE_PY = os.path.join(str(get_project_root()), "example_pipelines", "healthcare", "healthcare.py")


def check_exec_without_errors(filepath):
    """
    Tests whether the pipeline file works
    """
    with open(filepath, encoding="utf-8") as file:
        text = file.read()
        parsed_ast = ast.parse(text)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"), {})


def test_adult_complex_example_pipeline_runs():
    check_exec_without_errors(ADULT_COMPLEX_PY)


def test_amazon_reviews_example_pipeline_runs():
    check_exec_without_errors(AMAZON_REVIEWS_PY)


def test_compas_example_pipeline_runs():
    check_exec_without_errors(COMPAS_PY)


def test_healthcare_example_pipeline_runs():
    check_exec_without_errors(HEALTHCARE_PY)
