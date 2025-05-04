import streamlit as st
import streamlit.web.bootstrap
import streamlit.components.v1 as components
import pandas as pd

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

        # ------------------------------------------------------------------
        # 1️⃣ Make sure the legacy harmful column exists so we never lose it
        # ------------------------------------------------------------------
        if "harmful" not in df.columns:
            df["harmful"] = ""
        # ------------------------------------------------------------------

        # Display the DataFrame
        st.write("Original Data")
        st.dataframe(df)

        # Input for new column name
        new_column_name = st.text_input("Enter new column name")

        # Add new column if it doesn't exist
        if new_column_name and new_column_name not in df.columns:
            df[new_column_name] = ""

        # Annotate each row
        for index, row in df.iterrows():
            st.write(f"Row {index + 1}")
            st.write(f"Prompt: {row['prompt']}")
            st.write(f"Response: {row['response']}")

            # --------------------------------------------------------------
            # 2️⃣ Preserve & display existing ‘harmful’ annotation
            # --------------------------------------------------------------
            harmful_options = ["", "True", "False"]
            current_harmful_val = str(row.get("harmful", ""))
            if pd.isna(current_harmful_val) or current_harmful_val not in harmful_options:
                current_harmful_val = ""
            harmful = st.selectbox(
                "Harmful",
                options=harmful_options,
                index=harmful_options.index(current_harmful_val),
                key=f"{index}_harmful",
            )
            df.at[index, "harmful"] = harmful
            # --------------------------------------------------------------

            # --------------------------------------------------------------
            # 3️⃣ Preserve & display existing value for any *new* column
            # --------------------------------------------------------------
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
            # --------------------------------------------------------------

        # Display the annotated DataFrame
        st.write("Annotated Data")
        st.dataframe(df)

        # Save the annotated DataFrame
        if st.button("Save Annotations"):
            save_csv(df, "annotated_data.csv")
            st.success("Annotations saved to annotated_data.csv")

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap
        streamlit.web.bootstrap.run(__file__, False, [], {})
    main()
