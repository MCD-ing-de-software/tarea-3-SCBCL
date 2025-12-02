import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing."""
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

    # --- INICIO DE TESTS A COMPLETAR (A2) ---

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        """Test que verifica que drop_invalid_rows elimina filas con NaN/None."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        # Limpiar basado en 'name' y 'age'. 
        # Fila 1 (Bob) tiene age=None, Fila 2 (None) tiene name=None.
        result = cleaner.drop_invalid_rows(df, ["name", "age"])
        
        # Verificar que no hay nulos en las columnas seleccionadas
        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)
        
        # Verificar que el resultado tiene menos filas que el original
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        """Test que verifica que lanza KeyError para columnas inexistentes."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["columna_fantasma"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        """Test que verifica trim_strings limpia espacios sin alterar el DF original."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        # Guardar valor original para comparar
        original_value_alice = df.loc[0, "name"]  # " Alice "
        
        result = cleaner.trim_strings(df, ["name"])
        
        # 1. Verificar que el DF original NO fue modificado
        self.assertEqual(df.loc[0, "name"], original_value_alice)
        
        # 2. Verificar que en el resultado sí se limpió
        self.assertEqual(result.loc[0, "name"], "Alice")
        self.assertEqual(result.loc[3, "name"], "Carol")
        
        # 3. Verificar que columna 'city' (no tocada) es idéntica
        pdt.assert_series_equal(df["city"], result["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        """Test que verifica que lanza TypeError si la columna no es string."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        # 'age' es numérica, debería fallar
        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        """Test que verifica eliminación de outliers con IQR."""
        df = make_sample_df()
        cleaner = DataCleaner()
        
        # En make_sample_df, age tiene [25, None, 35, 120]. 120 es outlier.
        # Nota: remove_outliers_iqr usa dropna o cálculos que ignoran NaNs internamente para los cuantiles?
        # Revisando src/data_cleaner.py: usa .quantile() que ignora NaNs por defecto.
        
        result = cleaner.remove_outliers_iqr(df, "age", factor=1.5)
        
        # Verificar que el valor extremo (120) NO está en el resultado
        self.assertNotIn(120, result["age"].values)
        
        # Verificar que un valor normal (25) SÍ está
        self.assertIn(25, result["age"].values)

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
        
        # 'city' es texto
        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city")


if __name__ == "__main__":
    unittest.main()
