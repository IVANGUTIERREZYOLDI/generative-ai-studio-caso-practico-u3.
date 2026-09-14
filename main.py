"""
main.py — Aplicación de Generación de Imágenes y Edición de Contenido
Caso Práctico Unidad 3 · Generative AI · Instituto Europeo de Posgrado

Interfaz Streamlit con tres pantallas:
  1) Generación de imágenes (Stable Diffusion vía Bedrock)
  2) Edición de contenido (Claude vía Bedrock), con historial de versiones
  3) Colaboración: roles, permisos y comentarios

Todas las llamadas a los modelos pasan por bedrock_client.py (el "wrapper").
"""

import io
from datetime import datetime

import streamlit as st

import bedrock_client as bc

# --------------------------------------------------------------------------
# Configuración de página y estado de sesión
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Generative AI Studio — Marketing",
    page_icon="🎨",
    layout="wide",
)

def _init_state():
    defaults = {
        "gallery": [],          # lista de imágenes generadas (cada una es un diccionario)
        "text_history": [],     # lista de dicts: {version, texto, accion, usuario, ts}
        "current_text": "",
        "comments": [],         # lista de dicts: {autor, rol, texto, ts}
        "current_user": "Ana (Diseñadora)",
        "current_role": "Diseñador",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

ROLES = {
    "Ana (Diseñadora)": "Diseñador",
    "Luis (Redactor)": "Redactor",
    "Marta (Aprobadora)": "Aprobador",
}

ROLE_PERMISSIONS = {
    "Diseñador": {"Generación de imágenes"},
    "Redactor": {"Edición de contenido"},
    "Aprobador": {"Edición de contenido", "Colaboración"},
}
# Todos los roles pueden ver Colaboración para comentar; el admin ve todo.
ROLE_PERMISSIONS["Diseñador"].add("Colaboración")
ROLE_PERMISSIONS["Redactor"].add("Colaboración")


# --------------------------------------------------------------------------
# Barra lateral: identidad, rol y aviso de modo simulado
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🎨 Generative AI Studio")
    st.caption("Marketing & Publicidad — sobre Amazon Bedrock")

    user = st.selectbox("Usuario activo", list(ROLES.keys()),
                         index=list(ROLES.keys()).index(st.session_state.current_user))
    st.session_state.current_user = user
    st.session_state.current_role = ROLES[user]
    st.markdown(f"**Rol:** {st.session_state.current_role}")

    st.divider()
    if bc.USE_MOCK:
        st.warning(
            "**Modo simulado (mock).** No hay acceso a AWS Bedrock configurado. "
            "Las respuestas de Stable Diffusion y Claude se generan localmente "
            "para demostrar el flujo. El código real con `boto3` está listo en "
            "`bedrock_client.py` (comentado).",
            icon="⚠️",
        )
    else:
        st.success(f"Conectado a Amazon Bedrock (región {bc.AWS_REGION})", icon="✅")

    st.divider()
    st.caption("Modelos en uso")
    st.code(
        f"Imagen:  {bc.MODEL_ID_STABLE_DIFFUSION}\n"
        f"Texto:   {bc.MODEL_ID_CLAUDE}\n"
        f"Embed.:  {bc.MODEL_ID_TITAN_EMBEDDINGS}",
        language="text",
    )

allowed_tabs = ROLE_PERMISSIONS.get(st.session_state.current_role, set())

tab_titles = ["🖼️ Generación de imágenes", "✍️ Edición de contenido", "👥 Colaboración"]
tab_img, tab_text, tab_collab = st.tabs(tab_titles)


# --------------------------------------------------------------------------
# PANTALLA 1 — Generación de imágenes
# --------------------------------------------------------------------------

with tab_img:
    if "Generación de imágenes" not in allowed_tabs:
        st.info(f"Tu rol ({st.session_state.current_role}) no tiene acceso a esta "
                f"funcionalidad. Cambia a un usuario Diseñador en la barra lateral.")
    else:
        st.subheader("Generar imagen a partir de texto")
        col1, col2 = st.columns([2, 1])

        with col1:
            prompt = st.text_area(
                "Descripción de la imagen (prompt)",
                placeholder="Ej: un frasco de perfume flotando sobre fondo pastel, "
                            "luz suave, estética minimalista de campaña publicitaria",
                height=100,
            )
        with col2:
            style = st.selectbox("Estilo", list(bc.STYLE_PALETTES.keys()))
            seed_input = st.number_input("Semilla (seed) — opcional, 0 = aleatoria",
                                          min_value=0, value=0, step=1)
            generate_btn = st.button("🎨 Generar imagen", type="primary",
                                      use_container_width=True)

        if generate_btn:
            if not prompt.strip():
                st.error("Escribe una descripción antes de generar la imagen.")
            else:
                seed = None if seed_input == 0 else int(seed_input)
                with st.spinner("Generando imagen con Stable Diffusion (Bedrock)…"):
                    result = bc.generate_image(prompt, style, seed)
                st.session_state.gallery.insert(0, result)
                st.success(f"Imagen generada (seed={result['seed']}, estilo={style})")

        st.divider()
        st.subheader("📁 Galería")
        if not st.session_state.gallery:
            st.caption("Aún no se han generado imágenes en esta sesión.")
        else:
            cols = st.columns(3)
            for i, item in enumerate(st.session_state.gallery):
                with cols[i % 3]:
                    st.image(item["image"], use_container_width=True)
                    st.caption(f"**Estilo:** {item['style']} · **Seed:** {item['seed']}")
                    prompt_texto = item["prompt"]
                    if len(prompt_texto) > 80:
                        prompt_texto = prompt_texto[:80] + "…"
                    st.caption(prompt_texto)
                    buf = io.BytesIO()
                    item["image"].save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Descargar", data=buf.getvalue(),
                        file_name=f"imagen_{item['seed']}.png", mime="image/png",
                        key=f"dl_{i}_{item['seed']}",
                        use_container_width=True,
                    )


# --------------------------------------------------------------------------
# PANTALLA 2 — Edición de contenido (con historial de versiones)
# --------------------------------------------------------------------------

with tab_text:
    if "Edición de contenido" not in allowed_tabs:
        st.info(f"Tu rol ({st.session_state.current_role}) no tiene acceso a esta "
                f"funcionalidad. Cambia a un usuario Redactor o Aprobador.")
    else:
        st.subheader("Editar y mejorar contenido con Claude")

        original = st.text_area(
            "Texto a editar",
            value=st.session_state.current_text,
            placeholder="Pega aquí el texto publicitario o editorial que quieres mejorar…",
            height=160,
        )
        st.session_state.current_text = original

        action = st.radio(
            "Acción",
            list(bc.TEXT_TASK_PARAMS.keys()),
            horizontal=True,
        )
        params_preview = bc.TEXT_TASK_PARAMS[action]
        st.caption(
            f"Parámetros de inferencia → temperature={params_preview['temperature']} · "
            f"top_p={params_preview['top_p']} · max_tokens={params_preview['max_tokens']}"
        )

        if st.button("✨ Procesar con Claude", type="primary"):
            if not original.strip():
                st.error("Escribe o pega un texto antes de procesarlo.")
            else:
                with st.spinner(f"Aplicando '{action}' con Claude (Bedrock)…"):
                    result = bc.edit_text(original, action)
                version_n = len(st.session_state.text_history) + 1
                st.session_state.text_history.insert(0, {
                    "version": version_n,
                    "texto": result["text"],
                    "accion": action,
                    "usuario": st.session_state.current_user,
                    "ts": datetime.now().strftime("%H:%M:%S"),
                })
                st.session_state.current_text = result["text"]
                st.rerun()

        st.divider()
        st.subheader("🕓 Historial de versiones")
        if not st.session_state.text_history:
            st.caption("Aún no hay versiones generadas en esta sesión.")
        else:
            for h in st.session_state.text_history:
                with st.expander(
                    f"Versión {h['version']} · {h['accion']} · {h['usuario']} · {h['ts']}"
                ):
                    st.write(h["texto"])
                    if st.button("↩️ Restaurar esta versión", key=f"restore_{h['version']}"):
                        st.session_state.current_text = h["texto"]
                        st.rerun()


# --------------------------------------------------------------------------
# PANTALLA 3 — Colaboración: roles y comentarios
# --------------------------------------------------------------------------

with tab_collab:
    st.subheader("Equipo y roles")
    st.table(
        [{"Usuario": u, "Rol": r, "Acceso": ", ".join(sorted(ROLE_PERMISSIONS[r]))}
         for u, r in ROLES.items()]
    )

    st.divider()
    st.subheader("💬 Comentarios y notas")
    with st.form("comment_form", clear_on_submit=True):
        comment_text = st.text_area("Nuevo comentario", height=80)
        submitted = st.form_submit_button("Publicar comentario")
        if submitted and comment_text.strip():
            st.session_state.comments.insert(0, {
                "autor": st.session_state.current_user,
                "rol": st.session_state.current_role,
                "texto": comment_text.strip(),
                "ts": datetime.now().strftime("%d/%m %H:%M"),
            })

    if not st.session_state.comments:
        st.caption("Aún no hay comentarios.")
    else:
        for c in st.session_state.comments:
            st.markdown(f"**{c['autor']}** _( {c['rol']} )_ — {c['ts']}")
            st.write(c["texto"])
            st.markdown("---")
