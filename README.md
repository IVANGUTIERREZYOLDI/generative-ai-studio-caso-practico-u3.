# Generative AI Studio — Marketing & Publicidad

Aplicación web (Streamlit) para generación de imágenes y edición de contenido
mediante IA generativa, desarrollada como Caso Práctico de la Unidad 3
(Generative AI) del Instituto Europeo de Posgrado.

**Vía de entrega: Vía A — Aplicación funcional**, con simulación (mock) de
Amazon Bedrock por no disponer de una cuenta AWS con acceso a los modelos.
El código real con `boto3` está implementado y comentado en
`app/bedrock_client.py`, listo para activarse.

## Funcionalidades

- **Generación de imágenes**: texto → imagen (simulación de Stable Diffusion
  XL), selección de estilo (realismo, anime, óleo), semilla reproducible y
  galería descargable.
- **Edición de contenido**: resumir, corregir, expandir y generar variaciones
  de un texto (simulación de Claude), con historial de versiones y opción de
  restaurar una versión anterior.
- **Colaboración**: roles (diseñador, redactor, aprobador) con permisos
  distintos por pantalla, y sección de comentarios.

## Estructura del proyecto

```
proyecto/
├── app/
│   ├── main.py              # Interfaz Streamlit (3 pantallas)
│   ├── bedrock_client.py    # Wrapper de orquestación hacia Bedrock (mock + real)
│   └── requirements.txt
├── docs/
│   └── tronco_comun.md      # Diseño de la solución (secciones 3.1 a 3.6)
├── assets/
│   ├── arquitectura.png     # Diagrama de arquitectura
│   └── screenshots/         # Capturas de la app en funcionamiento (demo)
└── README.md
```

## Cómo ejecutar la aplicación

1. Instalar dependencias:
   ```bash
   pip install -r app/requirements.txt
   ```
2. Ejecutar la app:
   ```bash
   cd app
   streamlit run main.py
   ```
3. Abrir el navegador en `http://localhost:8501`.

Por defecto la aplicación arranca en **modo simulado** (`USE_MOCK = True` en
`bedrock_client.py`), por lo que no requiere credenciales de AWS para
funcionar y hacer la demo completa.

## Cómo activar Amazon Bedrock real

1. Tener una cuenta de AWS con acceso concedido a los modelos
   `stability.stable-diffusion-xl-v1`, `anthropic.claude-3-5-sonnet-*` y
   `amazon.titan-embed-text-v2` en la consola de Bedrock (**Model access**).
2. Configurar credenciales (`aws configure` o variables de entorno
   `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`).
3. Instalar `boto3` (ya incluido en `requirements.txt`).
4. En `app/bedrock_client.py`:
   - Cambiar `USE_MOCK = False`.
   - Descomentar el bloque "IMPLEMENTACIÓN REAL CON boto3" al final del
     archivo (funciones `_bedrock_invoke_stable_diffusion`,
     `_bedrock_invoke_claude`, `_bedrock_invoke_titan_embeddings`).
5. Ajustar `AWS_REGION` si la región de trabajo no es `us-east-1`.

No es necesario tocar `main.py`: la interfaz llama siempre a las mismas
funciones públicas (`generate_image`, `edit_text`, `embed_text`), sea cual
sea el modo activo.

## Documento de diseño (tronco común)

El diseño completo de la solución —historias de usuario, arquitectura,
modelos y parámetros, system prompt de Claude, decisión sobre RAG/memoria,
y plan de ética y seguridad— está en `docs/tronco_comun.md` y en el
documento Word entregado junto a este proyecto.

## Demostración

Las capturas comentadas de la aplicación en funcionamiento están en
`assets/screenshots/` y también incluidas en el documento entregado, como
alternativa al video (no se dispone de grabación de pantalla en este
entorno de desarrollo).

## Fuentes y herramientas utilizadas

- Documentación oficial de Amazon Bedrock (introducción, ingeniería de
  prompts, Anthropic Claude en Bedrock).
- Streamlit (framework de interfaz).
- Guía del trabajo práctico Unidad 3 — Instituto Europeo de Posgrado.
