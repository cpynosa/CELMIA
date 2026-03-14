import pandas as pd
import io

class DataProcessor:
    """Clase para procesar y analizar datasets"""
    
    def load_file(self, uploaded_file):
        """Carga archivo CSV o Excel"""
        
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Intentar diferentes encodings
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        else:
            raise ValueError(f"Formato no soportado: {file_extension}")
        
        return df
    
    def describe_dataset(self, df):
        """Genera una descripción completa del dataset"""
        
        description = []
        
        # Información básica
        description.append(f"INFORMACIÓN GENERAL:")
        description.append(f"- Total de filas: {df.shape[0]}")
        description.append(f"- Total de columnas: {df.shape[1]}")
        description.append(f"- Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Información de columnas
        description.append(f"\nCOLUMNAS:")
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            col_info = f"- {col}:"
            col_info += f" tipo={dtype},"
            col_info += f" únicos={unique_count},"
            col_info += f" nulos={null_count} ({null_pct:.1f}%)"
            
            # Agregar info específica según tipo
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info += f", rango=[{df[col].min():.2f}, {df[col].max():.2f}]"
            elif pd.api.types.is_string_dtype(df[col]):
                top_values = df[col].value_counts().head(3)
                col_info += f", top valores={top_values.index.tolist()}"
            
            description.append(col_info)
        
        # Tipos de datos
        description.append(f"\nTIPOS DE DATOS:")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if numeric_cols:
            description.append(f"- Numéricas ({len(numeric_cols)}): {', '.join(numeric_cols)}")
        if categorical_cols:
            description.append(f"- Categóricas ({len(categorical_cols)}): {', '.join(categorical_cols)}")
        if datetime_cols:
            description.append(f"- Fechas ({len(datetime_cols)}): {', '.join(datetime_cols)}")
        
        # Calidad de datos
        description.append(f"\nCALIDAD DE DATOS:")
        total_nulls = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        null_pct = (total_nulls / total_cells) * 100
        description.append(f"- Valores nulos totales: {total_nulls} ({null_pct:.2f}%)")
        
        duplicates = df.duplicated().sum()
        description.append(f"- Filas duplicadas: {duplicates}")
        
        return "\n".join(description)
    
    def get_column_types(self, df):
        """Clasifica columnas por tipo"""
        
        return {
            'numeric': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime': df.select_dtypes(include=['datetime']).columns.tolist(),
            'boolean': df.select_dtypes(include=['bool']).columns.tolist()
        }
    
    def suggest_analyses(self, df):
        """Sugiere análisis apropiados según el tipo de datos"""
        
        suggestions = []
        col_types = self.get_column_types(df)
        
        # Análisis numéricos
        if col_types['numeric']:
            suggestions.append({
                'type': 'numeric_summary',
                'description': 'Estadísticas descriptivas de variables numéricas',
                'columns': col_types['numeric']
            })
            
            if len(col_types['numeric']) >= 2:
                suggestions.append({
                    'type': 'correlation',
                    'description': 'Matriz de correlación entre variables numéricas',
                    'columns': col_types['numeric']
                })
        
        # Análisis categóricos
        if col_types['categorical']:
            suggestions.append({
                'type': 'frequency',
                'description': 'Distribución de frecuencias de variables categóricas',
                'columns': col_types['categorical']
            })
        
        # Análisis temporales
        if col_types['datetime']:
            suggestions.append({
                'type': 'time_series',
                'description': 'Análisis de series temporales',
                'columns': col_types['datetime']
            })
        
        # Análisis combinados
        if col_types['numeric'] and col_types['categorical']:
            suggestions.append({
                'type': 'grouped_analysis',
                'description': 'Análisis de variables numéricas agrupadas por categorías',
                'columns': {
                    'numeric': col_types['numeric'],
                    'categorical': col_types['categorical']
                }
            })
        
        return suggestions
    
    def clean_data(self, df, options=None):
        """Limpia el dataset según opciones especificadas"""
        
        df_clean = df.copy()
        
        if options is None:
            options = {
                'remove_duplicates': True,
                'fill_numeric_nulls': 'median',
                'fill_categorical_nulls': 'mode'
            }
        
        # Remover duplicados
        if options.get('remove_duplicates', False):
            df_clean = df_clean.drop_duplicates()
        
        # Manejar nulos en columnas numéricas
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        if options.get('fill_numeric_nulls'):
            method = options['fill_numeric_nulls']
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    if method == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif method == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif method == 'zero':
                        df_clean[col].fillna(0, inplace=True)
        
        # Manejar nulos en columnas categóricas
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        if options.get('fill_categorical_nulls'):
            method = options['fill_categorical_nulls']
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    if method == 'mode':
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col].fillna(mode_val[0], inplace=True)
                    elif method == 'unknown':
                        df_clean[col].fillna('Unknown', inplace=True)
        
        return df_clean
