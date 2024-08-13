# Proyecto: Framework milleLangChain
Este framework ha sido desarrollado a partir de LangChain con el objetivo de simplificar el proceso de uso de Modelos de Lenguaje Grande (LLMs), embeddings y almacenes vectoriales (vector stores), otros. Proporciona clases específicas para la invocación de LLMs, gestión de embeddings, creación y carga de vector stores, y un sistema de chat basado en LLMs.

## Características Principales
- LLMs: Clase general para la invocación de modelos de lenguaje grande.
- Embeddings: Clase para la gestión y generación de embeddings.
- Vector Store: Herramientas para la creación y carga de almacenes vectoriales, útiles en tareas como Recuperación de Información Asistida por Generación (RAG).
- ChatLLM: Clase para la creación de sistemas de chat basados en modelos de lenguaje, permitiendo la inferencia de LLMs en un entorno conversacional.
- LLMs Estructurados: Clase genericas para la creación de LLMs con tareas espeficicas

## Instalación
Para utilizar este framework, sigue los siguientes pasos:
1. Clona este repositorio:
```bash
https://github.com/CamiloP121/milleLangChain.git
```
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Ejmeplo de uso

### Crear embedding
```python
from milleLangChain.embbeding import embbeding
```

(Construcción...)