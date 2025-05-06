import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd

# -------------------------------------------------------------------
# Add as many classification columns as you like here ⤵
# -------------------------------------------------------------------
CLASSIFICATION_COLS = ["refusal", "ethical", "harmful"]
# -------------------------------------------------------------------

# Function to load CSV file
def load_csv(file):
    return pd.read_csv(file)

# Function to save the annotated DataFrame
def save_csv(df, filename):
    df.to_csv(filename, index=False)

# Streamlit app
def main():
    st.title("CSV Annotation Tool")

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load the CSV file
        df = load_csv(uploaded_file)

        # ---------------------------------------------------------------
        # Make sure every classification column exists
        # ---------------------------------------------------------------
        for col in CLASSIFICATION_COLS:
            if col not in df.columns:
                df[col] = ""

        # Display the DataFrame
        st.write("Original Data")
        st.dataframe(df)

        # Input for new column name
        new_column_name = st.text_input("Enter new column name")

        # Add new column if it doesn't exist
        if new_column_name and new_column_name not in df.columns:
            df[new_column_name] = ""

        # ---------------------------------------------------------------
        # Annotate each row
        # ---------------------------------------------------------------
        for index, row in df.iterrows():
            st.write(f"Row {index + 1}")
            st.write(f"Prompt: {row['prompt']}")
            st.write(f"Response: {row['response']}")

            # Iterate over every pre-defined classification column
            for col in CLASSIFICATION_COLS:
                options = ["", "True", "False"]
                current_val = str(row.get(col, ""))
                if pd.isna(current_val) or current_val not in options:
                    current_val = ""
                selected_val = st.selectbox(
                    col.capitalize(),                    # label
                    options=options,                    # "", True, False
                    index=options.index(current_val),   # preserve value
                    key=f"{index}_{col}",               # unique key
                )
                df.at[index, col] = selected_val

            # Handle the (optional) arbitrary new column
            if new_column_name:
                if pd.isna(df.at[index, new_column_name]):
                    df.at[index, new_column_name] = ""
                existing_val = str(df.at[index, new_column_name])
                new_value = st.text_input(
                    f"Enter value for {new_column_name}",
                    value=existing_val,
                    key=f"{index}_{new_column_name}",
                )
                df.at[index, new_column_name] = new_value

        # Display the annotated DataFrame
        st.write("Annotated Data")
        st.dataframe(df)

        # ------------------------------------------------------------------
        # 4️⃣  Save button moved to the sidebar
        # ------------------------------------------------------------------
        # Save the annotated DataFrame
        if st.sidebar.button("Save Annotations"):
            save_csv(df, "annotated_data.csv")
            st.sidebar.success("Annotations saved to annotated_data.csv")

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
