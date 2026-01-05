import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from rag import RAGPipeline
from utils import setup_logger

logger = setup_logger("evaluate")

# --- Manual Golden Dataset ---
GOLDEN_DATASET = [
    {
        "question": "Under the Fiancé/Fiancée Scheme, by when must the couple submit their marriage certificate to HDB?",
        "ground_truth": "Couples must submit their marriage certificate within 3 months of the resale completion date or the date of key collection for a new flat."
    },
    {
        "question": "What is the maximum gross monthly household income ceiling for a family to buy a 4-room or larger BTO flat under the Public Scheme?",
        "ground_truth": "The maximum gross monthly household income ceiling is $14,000 for families and $21,000 for extended/multi-generation families."
    },
    {
        "question": "At what age can a single Singapore Citizen apply for a 2-room Flexi BTO flat in any location?",
        "ground_truth": "Unmarried or divorced Singapore Citizens must be at least 35 years old. If widowed or an orphan, they can apply from age 21."
    },
    {
        "question": "What is the maximum amount of Enhanced CPF Housing Grant (EHG) a single person can receive when buying a flat on their own?",
        "ground_truth": "An eligible single applicant can receive up to $60,000 in Enhanced CPF Housing Grant (EHG) (Singles)."
    },
    {
        "question": "What are the lease options available for seniors (aged 55 and above) purchasing a 2-room Flexi flat?",
        "ground_truth": "Seniors can choose a lease between 15 and 45 years in 5-year increments, provided the lease covers the youngest owner until at least age 95."
    },
    {
        "question": "What is the minimum age requirement for all buyers and their spouses to qualify for a Community Care Apartment (CCA)?",
        "ground_truth": "All buyers and their spouses must be at least 65 years old at the time of application."
    },
    {
        "question": "Can a single Singapore Citizen buy a resale HDB flat of any size?",
        "ground_truth": "Yes, singles can buy a Standard or Plus resale flat of any size, except for 3Gen flats. For Prime resale flats, they are restricted to 2-room Flexi units."
    },
    {
        "question": "What is the purpose of the HDB Flat Eligibility (HFE) letter?",
        "ground_truth": "The HFE letter provides a consolidated assessment of a buyer's eligibility to purchase a flat, receive housing grants, and take up an HDB housing loan."
    },
    {
        "question": "What must an undischarged bankrupt obtain before applying for an HDB flat bigger than a 5-room flat?",
        "ground_truth": "Prior consent must be obtained from the Official Assignee (OA) or the private trustee."
    },
    {
        "question": "How long must a person wait after disposing of a private residential property before applying for an HFE letter to buy a subsidized flat?",
        "ground_truth": "They must wait at least 30 months from the legal completion date of the disposal."
    },
    {
        "question": "What document must all flat buyers have before they can apply for a flat in an HDB sales exercise?",
        "ground_truth": "All buyers must have a valid HDB Flat Eligibility (HFE) letter."
    },
    {
        "question": "Can seniors who own a non-residential property still buy a short-lease 2-room Flexi flat?",
        "ground_truth": "Yes, seniors aged 55 and above who own more than one non-residential property may buy a short-lease 2-room Flexi flat or Community Care Apartment."
    },
    {
        "question": "What is the total number of ballot chances for a First-Timer (Parents & Married Couples) applicant for a BTO flat?",
        "ground_truth": "FT(PMC) applicants receive a total of three ballot chances."
    },
    {
        "question": "Under the Seniors page, what is the minimum duration of the lease for a 2-room Flexi flat?",
        "ground_truth": "The minimum lease duration is 15 years."
    },
    {
        "question": "Can a permanent resident (PR) apply for a new HDB flat alone?",
        "ground_truth": "No, a PR must apply with at least one Singapore Citizen under the Public or Fiancé/Fiancée Scheme."
    },
    {
        "question": "Are singles allowed to buy 3Gen flats in the resale market?",
        "ground_truth": "No, singles are not eligible to buy 3Gen flats."
    },
    {
        "question": "What is the requirement for a child's custody to apply for a flat as a divorced parent?",
        "ground_truth": "The parent must have care and control of the child."
    }
]

JUDGE_PROMPT = """You are an expert judge for evaluating the quality of an answer given a ground truth.

    Question: {question}
    Ground Truth: {ground_truth}
    Actual Answer: {answer}

    Your task is to evaluate the Actual Answer against the Ground Truth based on two criteria:
    1. Core Facts: Does the Actual Answer contain the core facts found in the Ground Truth?
    2. No Contradiction: Does the Actual Answer avoid adding any contradicting information?

    If BOTH criteria are met, the verdict is PASS.
    If ANY criteria is not met, the verdict is FAIL.

    Explain clearly the reasoning before giving the verdict.

    Provide your evaluation in the following format:
    Reasoning: [Detailed explanation of your analysis]
    Verdict: [PASS or FAIL]
"""

def evaluate_with_llm_judge(llm, question, answer, ground_truth):
    logger.info(f"Processing llm judge: {question}")
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer
        })
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        reasoning = "N/A"
        verdict = "FAIL" # Default to fail if parsing error
        
        lines = content.strip().split('\n')
        for line in lines:
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Verdict:"):
                verdict = line.replace("Verdict:", "").strip()
                
        # structured correctly line-by-line
        if reasoning == "N/A" and "Verdict:" in content:
            parts = content.split("Verdict:")
            reasoning = parts[0].replace("Reasoning:", "").strip()
            verdict = parts[1].strip()
            
        return verdict, reasoning
        
    except Exception as e:
        logger.error(f"Error in LLM Judge: {e}")
        return "ERROR", f"Exception during evaluation: {e}"

def run_evaluation():
    logger.info("Initializing RAG for Evaluation...")
    rag = RAGPipeline()
    
    results_list = []
    
    logger.info("Running Generation and Evaluation on Golden Dataset (LLM Judge) ...")
    target_dataset = GOLDEN_DATASET
    
    for i, item in enumerate(target_dataset):
        q = item["question"]
        gt = item["ground_truth"]
        
        logger.info(f"Processing ({i+1}/{len(target_dataset)}): {q}")
        
        try:
            # RAG
            response, docs = rag.query(q)
            
            # LLM Judge
            verdict, reasoning = evaluate_with_llm_judge(rag.llm, q, response, gt)
            
            logger.info(f"Verdict: {verdict}")
            
            # Result
            results_list.append({
                "question": q,
                "ground_truth": gt,
                "answer": response,
                "verdict": verdict,
                "reasoning": reasoning,
                "contexts": [d.page_content for d in docs]
            })
            
        except Exception as e:
            logger.error(f"Error processing item {i}: {e}")
            continue

    if results_list:
        print("\n=== Evaluation Results ===")
        df = pd.DataFrame(results_list)
        print(df[["question", "ground_truth", "answer", "verdict", "reasoning"]])
        
        output_file = "evaluation_results_llm_judge.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    run_evaluation()
