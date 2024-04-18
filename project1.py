import streamlit as st
from langchain_helper import create_vector_db,qa_chain
st.title("Bioactivity QA ⚕️💊")
btn= st.button("Create Knowledgebase")
if btn:
    pass
question=st.text_input("Question: ")
example_questions = [
    "What is the pIC50 value of compound whose canonical smile is CCOc1nn(-c2cccc(OCc3ccccc3)c2)c(=O)o1 and also calculates its drug likeness parameter?(6.1249387366083)",
    "Give me pic50 value of canonical smile O=C(N1CCCCC1)n1nc(-c2ccc(Cl)cc2)nc1SCC(F)(F)F and properties that lean towards good oral bioavailability.(6.523)",
    "what is the pIC50 value of CSc1nc(-c2ccc(-c3ccccc3)cc2)nn1C(=O)N(C)C canonical smile exhibits a unique property due to the presence of the thioamide group(6.251812)",
    "what is the pic50 value of C[C@H]1C(=O)N(C(=O)NCc2ccccc2)[C@@H]1Oc1ccc(C(=O)C(C)(C)C)cc1 canonical smile compound which includes chirality indicators?(4)"
]
st.sidebar.title("Example Questions")

# Display example questions in the sidebar
for q in example_questions:
    st.sidebar.markdown(f"- {q}")

if question:
    chain=qa_chain()
    response=chain(question)

    st.header("Answer: ")
    st.write(response["result"])
