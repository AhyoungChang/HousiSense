"""Streamlit interface for HousiSense v2.

Run the scripts in scripts/ first to populate PostGIS, then:
    streamlit run app.py
"""

import json

import folium
import streamlit as st
from streamlit_folium import st_folium

from housisense import search

AUSTIN = [30.2672, -97.7431]

EXAMPLE = "quiet place near a park, walkable to lots of cafes"


def draw_map(ranked):
    m = folium.Map(location=AUSTIN, zoom_start=12)
    for position, (_, row) in enumerate(ranked.iterrows(), 1):
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=f"#{position} {row['name']} (${row['price']})",
            icon=folium.Icon(color="red" if position == 1 else "blue", icon="home"),
        ).add_to(m)
    return m


def main():
    st.set_page_config(layout="wide", page_title="HousiSense v2")
    left, right = st.columns(2)

    with left:
        st.subheader("Chat")
        query = st.chat_input(f"e.g. {EXAMPLE}")
        if query:
            with st.spinner("planning and searching"):
                plan, ranked, explanations, relaxed = search.run(query)

            st.caption("plan: " + json.dumps(plan, ensure_ascii=False))
            if relaxed:
                st.info("These constraints matched nothing and were dropped: "
                        + json.dumps(relaxed, ensure_ascii=False))
            if ranked.empty:
                st.warning("No listing satisfies even one constraint. Try a looser query.")

            for (_, row), explanation in zip(ranked.iterrows(), explanations):
                st.markdown(f"**{row['name']}** (${row['price']}) — {explanation}")

            st.session_state["map"] = draw_map(ranked)

    with right:
        st.subheader("Map")
        st_folium(
            st.session_state.get("map", folium.Map(location=AUSTIN, zoom_start=11)),
            width="100%", height=600, returned_objects=[],
        )


if __name__ == "__main__":
    main()
