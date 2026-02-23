import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import pandas as pd

from our_own_rag_system.intelligent_rag import IntelligentRAG

# Load evaluator functions.
from evaluation.evalute_rag import RAGEvaluator, create_test_cases
from generate_rag_report import RAGReportGenerator

class evaluate_rag_techniques:
    def __init__(self, file_path:str, question_file_path:str, metric_list:list[str]):
        self.file_path = file_path
        self.question_file_path = question_file_path
        self.questions = []
        self.ground_truth = []
        self.load_questions()
        self.metric_list = metric_list
        self.evaluator = RAGEvaluator()

        self.rag_techniques = {
            "intelligent_rag" : IntelligentRAG
        }

    
    def load_questions(self):
        json_file = ""
        with open(self.question_file_path, 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        
        for idx, data  in json_file.items():
            self.questions.append(data['question'])
            self.ground_truth.append(data['ground_truth'])


    def _evaluate_rag_technique(self, rag_technique:str):
        rag_llm_outputs = []
        rag_llm_contexts = []
        for question, ground_truth in zip(self.questions, self.ground_truth):
            rag_llm_output, rag_llm_context = rag_technique.query(question=question)
            rag_llm_outputs.append(rag_llm_output)
            rag_llm_contexts.append(rag_llm_context)

        rag_test_cases = create_test_cases(
            questions=self.questions,
            generated_answers=rag_llm_outputs,
            ground_truths=self.ground_truth,
            retrieved_contexts=rag_llm_contexts
        )

        rag_report = self.evaluator.evaluate_batch(rag_test_cases, metrics=self.metric_list)

        return json.dumps(rag_report.to_dict(), indent=4, default=str)
    
    def evaluate(self, pdf_output_path:str="rag_techniques_output.pdf", json_output_path:str="rag_techniques_output.json"):
        context_chunk_report = []
        hyde_report = []
        hyper_report = []
        query_transform_report = []
        reliable_report = []
        proposition_chunk_report = []
        simple_report = []

        ## context chunk header rag
        # context_chunk_llm_outputs = []
        # context_chunk_llm_contexts = []
        # for question, ground_truth in zip(self.questions, self.ground_truth):
        #     for metric in self.metric_list:
        #         context_chunk_llm_output, context_chunk_llm_context = self.context_chunk_rag.query(question=question)
        #         context_chunk_llm_outputs.append(context_chunk_llm_output)
        #         context_chunk_llm_contexts.append(context_chunk_llm_context)

        # rag_test_cases = create_test_cases(
        #     questions=self.questions,
        #     generated_answers=context_chunk_llm_outputs,
        #     ground_truths=self.ground_truth,
        #     retrieved_contexts=context_chunk_llm_contexts
        # )

        # context_chunk_report = self.evaluator.evaluate(rag_test_cases, metrics=self.metric_list)

        # json.dump(context_chunk_report.to_dict(), open("context_chunk_report.json", "w"), indent=4, default=str)
        
        output_dict = {}

        for name, rt in self.rag_techniques.items():
            print(f"start working on this technique: {name}")
            rt_obj = rt(file_path=self.file_path)
            
            tech_output = self._evaluate_rag_technique(rt_obj)
            output_dict[name] = tech_output

        with open(json_output_path, "w", encoding='utf-8') as f:
            json.dump(output_dict, f, indent=4, default=str)

        pdf_report_generator = RAGReportGenerator(json_input_filepath=json_output_path, pdf_output_filepath=pdf_output_path)
        pdf_report_generator.generate()



if __name__ == "__main__":
    
    demo_evaluate_rag_techniques = evaluate_rag_techniques(file_path=r"data\IntermediaryGuidelinesandDigitalMediaEthicsCode.pdf", question_file_path=r"data\gold_standared_q_a.json", metric_list=["correctness", "faithfulness", "relevancy", "completeness"])

    jsonfile_output_path = "own_rag_techniques_output.json"
    pdf_output_path = "own_rag_techniques_output.pdf"

    demo_evaluate_rag_techniques.evaluate(json_output_path=jsonfile_output_path, pdf_output_path=pdf_output_path)