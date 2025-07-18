import streamlit as st
import pandas as pd
import streamlit as st
import difflib



@st.cache_data
def load_data():
    df = pd.read_pickle("exports/filtered_patient_reports_with_event_log/combined.pkl")
    return df.applymap(str)

def main():
    st.title("Patient Reports Viewer")

    df = load_data()
    display_cols = ["HADM_ID","CHARTTIME","TEXT"]

    # Add a checkbox column for user interaction
    display_df = df[display_cols].copy()
    display_df["View"] = False  # Initially all unchecked

    st.subheader("Select rows to view full details")
    edited_df = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
    )

    # Filter rows where "View" is True
    selected_rows = edited_df[edited_df["View"] == True]
    if len(selected_rows) > 2:
        st.warning("Please select at most 2 rows to view details.")

    if not selected_rows.empty:
        st.subheader("Selected Rows")
        for idx in selected_rows.index:
            st.markdown("---")
            st.markdown(f"### Details for Index: {idx}")
            st.json(df.loc[idx].to_dict())
            st.markdown("---")
            for k,v in df.loc[idx].to_dict().items():
                if k == "Events":
                    st.write(v)
            # st.write()
    else:
        st.info("Check the 'View' box in any row to see details.")
    
    
    
    if len(selected_rows) == 2:
        highlight_differences(selected_rows['TEXT'].iloc[0], selected_rows['TEXT'].iloc[1])

def highlight_differences(text1: str, text2: str):
    """
    Display side-by-side differences between two strings with highlights.
    """
    differ = difflib.HtmlDiff()
    html_diff = differ.make_table(
        text1.splitlines(), 
        text2.splitlines(), 
        context=True, 
        numlines=2
    )
    
    with open("exports/string1.txt", "w", encoding="utf-8") as f:
        f.write(text1)
        
    with open("exports/string2.txt", "w", encoding="utf-8") as f:
        f.write(text2)

    st.markdown("### Differences Between Strings")
    st.components.v1.html(
        f"<style>table.diff {{font-family: monospace; font-size: 0.9em}}</style>{html_diff}",
        height=400,
        scrolling=True,
    )

if __name__ == "__main__":
    main()
