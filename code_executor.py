import sys
import io
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr

class CodeExecutor:
    """Ejecuta código Python de forma controlada"""
    
    def __init__(self):
        self.allowed_imports = [
            'pandas', 'numpy', 'matplotlib', 'seaborn', 
            'scipy', 'sklearn', 'math', 'datetime',
            'collections', 'itertools', 'statistics'
        ]
    
    def execute_code(self, code, df):
        """Ejecuta código Python y captura resultados"""
        
        results = {
            'success': False,
            'output': '',
            'error': None,
            'plots': [],
            'code_used': code  # Guardar el código que intentamos ejecutar
        }
        
        # Sanitizar código antes de ejecutar
        code = self.sanitize_code(code)
        results['code_used'] = code  # Actualizar con versión sanitizada
        
        # Validar sintaxis antes de ejecutar
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            results['error'] = f"Error de sintaxis en línea {e.lineno}: {e.msg}"
            results['error'] += f"\n\nCódigo problemático:\n{self._get_error_context(code, e.lineno)}"
            return results
        
        # Preparar el ambiente de ejecución
        namespace = {
            'df': df,
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'plt': plt,
            'sns': __import__('seaborn'),
            'plots': []
        }
        
        # Capturar stdout
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            # Ejecutar código
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                exec(code, namespace)
            
            # Capturar plots generados
            if 'plots' in namespace and namespace['plots']:
                results['plots'] = namespace['plots']
            else:
                # Si no usaron la lista plots, intentar capturar figuras abiertas
                figs = [plt.figure(i) for i in plt.get_fignums()]
                results['plots'] = figs
            
            results['success'] = True
            results['output'] = output_buffer.getvalue()
            
            # Si hubo errores pero el código se ejecutó
            if error_buffer.getvalue():
                results['warnings'] = error_buffer.getvalue()
        
        except Exception as e:
            results['success'] = False
            results['error'] = f"{type(e).__name__}: {str(e)}"
            results['output'] = output_buffer.getvalue()
            
            # Intentar dar contexto del error
            if error_buffer.getvalue():
                results['error'] += f"\n\nDetalles: {error_buffer.getvalue()}"
        
        finally:
            # Limpiar plots para evitar acumulación
            plt.close('all')
        
        return results
    
    def _get_error_context(self, code, line_num, context_lines=3):
        """Obtiene el contexto alrededor de una línea con error"""
        lines = code.split('\n')
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        
        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            context.append(f"{prefix}{i+1}: {lines[i]}")
        
        return '\n'.join(context)
    
    def validate_code(self, code):
        """Valida que el código sea seguro antes de ejecutarlo"""
        
        dangerous_patterns = [
            'import os',
            'import subprocess',
            'import sys',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code.lower():
                return False, f"Código contiene patrón no permitido: {pattern}"
        
        return True, "Código validado"
    
    def sanitize_code(self, code):
        """Limpia y prepara el código para ejecución"""
        
        # Remover bloques de markdown si existen
        if '```python' in code:
            code = code.split('```python')[1]
        if '```' in code:
            code = code.split('```')[0]
        
        # Remover imports peligrosos
        safe_code = code.strip()
        
        # Correcciones comunes de matplotlib
        replacements = {
            'plt.show()': '# plt.show() removido',
            'fig.title(': 'plt.title(',
            'fig.xlabel(': 'plt.xlabel(',
            'fig.ylabel(': 'plt.ylabel(',
            'ax.title(': 'ax.set_title(',
            'ax.xlabel(': 'ax.set_xlabel(',
            'ax.ylabel(': 'ax.set_ylabel(',
        }
        
        for old, new in replacements.items():
            safe_code = safe_code.replace(old, new)
        
        # Limpiar líneas vacías múltiples
        lines = safe_code.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            # Remover líneas con solo espacios
            stripped = line.strip()
            
            # Remover comentarios que el LLM pudo haber agregado
            if stripped.startswith('#') and 'import' in stripped.lower():
                continue
            
            # Evitar múltiples líneas vacías consecutivas
            if not stripped:
                if not prev_empty:
                    cleaned_lines.append(line)
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        safe_code = '\n'.join(cleaned_lines)
        
        # Agregar manejo de plots si no existe
        if 'plots' not in safe_code and 'plt.' in safe_code:
            # Intentar agregar guardado de plots automático
            lines = safe_code.split('\n')
            modified_lines = []
            
            for i, line in enumerate(lines):
                modified_lines.append(line)
                # Si crea una figura, guardarla
                if 'plt.figure' in line or ('fig' in line and '=' in line and 'plt' in line):
                    # Agregar con la misma indentación
                    indent = len(line) - len(line.lstrip())
                    modified_lines.append(' ' * indent + 'plots.append(plt.gcf())')
        
            safe_code = '\n'.join(modified_lines)
        
        return safe_code
    
    def get_code_summary(self, code):
        """Genera un resumen de lo que hace el código"""
        
        summary = []
        
        if 'import' in code:
            summary.append("Importa librerías necesarias")
        
        if 'describe()' in code:
            summary.append("Calcula estadísticas descriptivas")
        
        if 'corr()' in code:
            summary.append("Analiza correlaciones")
        
        if 'groupby' in code:
            summary.append("Agrupa datos por categorías")
        
        if 'plot' in code or 'plt.' in code:
            summary.append("Genera visualizaciones")
        
        if 'hist' in code:
            summary.append("Crea histogramas")
        
        if 'scatter' in code:
            summary.append("Crea gráficos de dispersión")
        
        if 'bar' in code:
            summary.append("Crea gráficos de barras")
        
        if not summary:
            summary.append("Realiza análisis personalizado")
        
        return " | ".join(summary)
