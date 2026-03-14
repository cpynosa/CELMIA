import requests
import json

class LLMAnalyzer:
    """Clase para interactuar con Ollama y generar análisis"""
    
    def __init__(self, model="qwen2.5-coder:7b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def check_ollama(self):
        """Verifica si Ollama está corriendo"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def query_llm(self, prompt, temperature=0.3):
        """Consulta al LLM local"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code}"
        
        except Exception as e:
            return f"Error al conectar con Ollama: {str(e)}"
    
    def generate_analysis_plan(self, dataset_description):
        """Genera un plan de análisis basado en la descripción del dataset"""
        
        prompt = f"""Eres un experto analista de datos. Basándote en la siguiente descripción de 
        un dataset, genera un plan de análisis conciso y accionable.

DESCRIPCIÓN DEL DATASET:
{dataset_description}

INSTRUCCIONES:
- Identifica los análisis más relevantes para este tipo de datos
- Sugiere visualizaciones apropiadas
- Menciona posibles correlaciones o patrones a buscar
- Sé específico pero conciso (máximo 200 palabras)
- Escribe en español

PLAN DE ANÁLISIS:"""

        return self.query_llm(prompt, temperature=0.5)
    
    def generate_analysis_code(self, df, dataset_description):
        """Genera código Python para analizar el dataset"""
        
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].head(3).tolist()
            columns_info.append(f"- {col} ({dtype}): ejemplos = {sample_values}")
        
        columns_text = "\n".join(columns_info)
        
        prompt = f"""Eres un programador experto en análisis de datos con Python. Genera código Python limpio y funcional para analizar este dataset.

INFORMACIÓN DEL DATASET:
- Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas
- Columnas:
{columns_text}

REGLAS CRÍTICAS DE SINTAXIS:
1. Usa SOLO estas librerías: pandas (pd), numpy (np), matplotlib.pyplot (plt)
2. El DataFrame YA está cargado en la variable 'df'
3. Ya existe una lista vacía 'plots = []'
4. NO escribas import statements
5. NO uses comillas triples (''') en el código
6. Verifica que todas las llaves, paréntesis y corchetes estén balanceados
7. Escribe código que funcione sin errores de sintaxis
8. Realiza visualizaciones claras, evitando gráficos muy complejos o innecesarios
9. Realiza gráficos con grandes insights, no gráficos decorativos o muy simples

REGLAS DE MATPLOTLIB:
- Crear figura: plt.figure(figsize=(10, 6))
- Títulos: plt.title("texto"), plt.xlabel("texto"), plt.ylabel("texto")
- NO usar: fig.title(), fig.xlabel(), fig.ylabel() (dan error)
- Con subplots: ax.set_title(), ax.set_xlabel(), ax.set_ylabel()
- NO usar plt.show()
- Después de crear una visualización: plots.append(plt.gcf())
- Usar plt.tight_layout() antes de guardar

ESTRUCTURA REQUERIDA:
```
# Estadísticas básicas
print("=== ANÁLISIS DESCRIPTIVO ===")
print(df.describe())
print()

# Análisis específico según tipo de datos
# ... tu código aquí ...

# Visualización 1
plt.figure(figsize=(10, 6))
# ... código de visualización ...
plt.title("Título Descriptivo")
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.tight_layout()
plots.append(plt.gcf())

# Visualización 2 (si aplica)
plt.figure(figsize=(10, 6))
# ... código de visualización ...
plt.title("Título Descriptivo 2")
plt.tight_layout()
plots.append(plt.gcf())
```

IMPORTANTE:
- Genera código SIMPLE y PROBADO
- Prefiere .head(), .describe(), .value_counts() sobre análisis complejos
- Si hay columnas numéricas, crea un histograma o boxplot
- Si hay columnas categóricas, crea un countplot o barplot
- Máximo 2-3 visualizaciones

Genera SOLO el código Python funcional, sin explicaciones ni markdown:"""

        code = self.query_llm(prompt, temperature=0.1)  # Temperatura más baja para más consistencia
        
        # Limpiar el código de forma más agresiva
        code = code.strip()
        
        # Remover múltiples tipos de bloques markdown
        if '```python' in code:
            parts = code.split('```python')
            if len(parts) > 1:
                code = parts[1]
        elif '```' in code:
            parts = code.split('```')
            if len(parts) > 1:
                code = parts[1] if len(parts) == 3 else parts[0]
        
        # Remover trailing ```
        if code.endswith('```'):
            code = code[:-3]
        
        # Remover líneas con solo espacios
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(lines)
        
        return code.strip()
    
    def generate_insights(self, dataset_description, execution_results):
        """Genera insights en lenguaje natural basados en los resultados"""
        
        output_text = execution_results.get('output', 'No hay salida de texto')
        
        prompt = f"""Eres un analista de datos experto. Basándote en los siguientes resultados de análisis, genera insights valiosos y accionables.

DESCRIPCIÓN DEL DATASET:
{dataset_description}

RESULTADOS DEL ANÁLISIS:
{output_text}

INSTRUCCIONES:
- Identifica los 3-5 insights más importantes
- Explica qué significan para el negocio/investigación
- Sugiere acciones concretas basadas en los datos
- Usa lenguaje claro y no técnico
- Escribe estrictamente en español
- Usa bullets points (-)
- Máximo 250 palabras

INSIGHTS:"""

        insights = self.query_llm(prompt, temperature=0.6)
        return insights
    
    def explain_visualization(self, plot_description):
        """Explica una visualización en lenguaje simple"""
        
        prompt = f"""Explica esta visualización en términos simples para alguien sin conocimientos técnicos:

{plot_description}

Usa máximo 2 frases. Escribe en español."""

        return self.query_llm(prompt, temperature=0.4)