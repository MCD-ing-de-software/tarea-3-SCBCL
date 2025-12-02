import pandas as pd
import pandas.testing as pdt
import unittest
import numpy as np

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    The DataFrame intentionally contains missing values, extra whitespace
    in a text column, and an obvious numeric outlier.
    """
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar DataFrames completos."""
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()
        
        result = cleaner.trim_strings(df, ["name"])
        
        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })
        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        """Ejemplo de test usando pandas.testing para comparar Series."""
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name"])
        
        # Los índices después de drop_invalid_rows son [0, 2]
        expected_name_series = pd.Series(["Alice", "Bob"], index=[0, 2], name="name")
        pdt.assert_series_equal(result["name"], expected_name_series, check_names=True)

    # --- TESTS COMPLETADOS ---

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que drop_invalid_rows elimina filas con NaN/None."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        result = cleaner.drop_invalid_rows(df, ["name", "age"])
        
        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que lanza KeyError para columnas inexistentes."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["columna_fantasma"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica trim_strings limpia espacios sin alterar el DF original."""
        # Se usa un DF sin nulos para evitar conflictos de tipo en esta prueba específica
        df = pd.DataFrame({
            "name": [" Alice ", "Bob", " Carol  "],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()
        
        original_value_alice = df.loc[0, "name"]
        
        result = cleaner.trim_strings(df, ["name"])
        
        # 1. Verificar original intacto
        self.assertEqual(df.loc[0, "name"], original_value_alice)
        
        # 2. Verificar limpieza
        self.assertEqual(result.loc[0, "name"], "Alice")
        self.assertEqual(result.loc[2, "name"], "Carol")
        
        # 3. Verificar otras columnas intactas
        pdt.assert_series_equal(df["city"], result["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que lanza TypeError si la columna no es string."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica eliminación de outliers con IQR."""
        # Se usa un set de datos estadísticamente claro
        df = pd.DataFrame({
            "val": [20, 21, 19, 20, 22, 1000]
        })
        cleaner = DataCleaner()
        
        result = cleaner.remove_outliers_iqr(df, "val", factor=1.5)
        
        # Verificar eliminación del outlier (1000)
        self.assertNotIn(1000, result["val"].values)
        
        # Verificar conservación de valores normales
        self.assertIn(20, result["val"].values)

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        """Test que verifica que lanza KeyError si falta la columna."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salario_inexistente")

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que lanza TypeError si la columna no es numérica."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")


if __name__ == "__main__":
    unittest.main()
