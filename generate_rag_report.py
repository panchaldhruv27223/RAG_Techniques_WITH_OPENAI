
import json
import os
import argparse
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGReportGenerator:
    """
    Generates a professional PDF report from RAG evaluation JSON output.
    """
    
    def __init__(self, json_input_filepath: str, pdf_output_filepath: str):
        self.input_path = json_input_filepath
        self.output_path = pdf_output_filepath
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='WinnerTitle',
            parent=self.styles['Heading2'],
            textColor=colors.darkgreen,
            alignment=1  # Center
        ))
        self.styles.add(ParagraphStyle(
            name='DetailLabel',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            spaceAfter=2
        ))
        self.styles.add(ParagraphStyle(
            name='DetailText',
            parent=self.styles['Normal'],
            spaceAfter=8,
            leftIndent=12
        ))
        self.styles.add(ParagraphStyle(
            name='ContextText',
            parent=self.styles['Normal'],
            fontSize=8,
            spaceAfter=4,
            leftIndent=12,
            textColor=colors.darkslategrey
        ))

    def load_data(self) -> Dict[str, Any]:
        """Loads and parses the JSON data."""
        logger.info(f"Loading data from {self.input_path}")
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in input file: {self.input_path}")
            raise

        parsed_data = {}
        for technique, result_str in raw_data.items():
            if isinstance(result_str, str):
                try:
                    parsed_data[technique] = json.loads(result_str)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON string for technique: {technique}")
                    parsed_data[technique] = {"error": "Failed to parse JSON string"}
            else:
                parsed_data[technique] = result_str
        
        return parsed_data

    def _get_metric_val(self, metrics: Dict, key: str) -> float:
        """Helper to safely get a metric value."""
        return metrics.get("averages", {}).get(key, 0) or 0

    def _get_all_metric_names(self, data: Dict[str, Any]) -> List[str]:
        """Dynamically discover all metric names from the evaluation data."""
        for metrics in data.values():
            averages = metrics.get("averages", {})
            if averages:
                return list(averages.keys())
        # Fallback if no averages found
        return ["correctness", "faithfulness", "relevancy", "context_precision"]

    def _identify_winner(self, data: Dict[str, Any]) -> str:
        """Identifies the 'winning' technique based on average of ALL available metrics."""
        metric_names = self._get_all_metric_names(data)
        best_tech = None
        best_score = -1.0

        for tech, metrics in data.items():
            if "error" in metrics:
                continue

            scores = [self._get_metric_val(metrics, m) for m in metric_names]
            avg_score = sum(scores) / len(scores) if scores else 0.0

            if avg_score > best_score:
                best_score = avg_score
                best_tech = tech

        logger.info(f"identified winner: {best_tech} with avg score: {best_score:.4f}")
        return best_tech

    def _create_summary_table(self, data: Dict[str, Any]) -> Table:
        """Creates the comparative summary table with dynamic metrics."""
        metric_names = self._get_all_metric_names(data)

        # Build dynamic headers: Technique | Metric1 | Metric2 | ... | Time (s)
        headers = ["Technique"] + [m.replace("_", " ").title() for m in metric_names] + ["Time (s)"]
        table_data = [headers]

        # Sort by first metric descending
        primary_metric = metric_names[0] if metric_names else "correctness"
        sorted_items = sorted(
            data.items(),
            key=lambda item: self._get_metric_val(item[1], primary_metric),
            reverse=True
        )

        for technique, metrics in sorted_items:
            if "error" in metrics:
                continue

            row = [technique.replace("_", " ").title()]
            row += [f"{self._get_metric_val(metrics, m):.2f}" for m in metric_names]
            row += [f"{metrics.get('elapsed_seconds', 0):.2f}"]
            table_data.append(row)

        # Dynamic column widths: first col wider, rest equal
        n_metric_cols = len(metric_names)
        col_widths = [1.5 * inch] + [0.9 * inch] * n_metric_cols + [0.9 * inch]
        table = Table(table_data, colWidths=col_widths)

        # Style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ])

        # Highlight best score per column (green = best)
        # Last column is Time (s) — lower is better; all others higher is better
        num_rows = len(table_data)
        time_col_idx = len(headers) - 1  # last column

        if num_rows > 1:
            for col_idx in range(1, len(headers)):
                values = []
                for r in range(1, num_rows):
                    try:
                        values.append(float(table_data[r][col_idx]))
                    except ValueError:
                        values.append(-1)

                if not values:
                    continue

                best_val = min(values) if col_idx == time_col_idx else max(values)

                for r in range(1, num_rows):
                    try:
                        if float(table_data[r][col_idx]) == best_val:
                            style.add('BACKGROUND', (col_idx, r), (col_idx, r), colors.lightgreen)
                    except ValueError:
                        pass

        table.setStyle(style)
        return table

    def _create_winner_detail_section(self, winner_name: str, winner_data: Dict[str, Any]) -> List:
        """Generates detailed view for the winning technique."""
        elements = []
        winner_display = winner_name.replace("_", " ").title()
        
        elements.append(PageBreak())
        elements.append(Paragraph(f"Winning Technique Analysis: {winner_display}", self.styles['WinnerTitle']))
        elements.append(Spacer(1, 12))
        
        results = winner_data.get("results", [])
        
        if not results:
            elements.append(Paragraph("No detailed results available.", self.styles['Normal']))
            return elements

        for i, res in enumerate(results):
            # Container for keeping Q&A together
            case_elements = []
            
            # Header
            case_elements.append(Paragraph(f"Case #{i+1}", self.styles['Heading3']))
            
            # Question
            case_elements.append(Paragraph("Question:", self.styles['DetailLabel']))
            case_elements.append(Paragraph(res.get("input_user_query", "N/A"), self.styles['DetailText']))
            
            # Actual Output
            case_elements.append(Paragraph("Model Answer:", self.styles['DetailLabel']))
            case_elements.append(Paragraph(res.get("actual_output", "N/A"), self.styles['DetailText']))
            
            # Context (Chunks)
            metrics_data = res.get("metrics", {})
            
            # Check context from metrics details if available, or try to infer from structure
            # The structure in json seems to put retrieval context in `metrics` -> `faithfulness` -> `details` -> `references` ?
            # Or usually it's passed in the input. Let's look at the Context Precision or Faithfulness details if available
            # Note: The provided JSON sample didn't explicitly show the 'retrieval_context' list in the `results` object directly,
            # but usually it's good to have. If not, we might be limited.
            # However, `faithfulness` metric details often contain formatted references.
            
            # Trying to extract context from Faithfulness/Relevance details if present
            context_text = "Context details not present in standard report output."
            
            # Let's try to extract from faithfulness details if available
            faith_details = metrics_data.get("faithfulness", {}).get("details", {})
            # This part depends on how `RAGEvaluator` logs details. 
            
            # Metrics Row
            metrics_str = []
            for m_name, m_data in metrics_data.items():
                score = m_data.get("score", 0)
                passed = "PASS" if m_data.get("passed") else "FAIL"
                color = "green" if m_data.get("passed") else "red"
                metrics_str.append(f"<b>{m_name.title()}:</b> <font color={color}>{score} ({passed})</font>")
            
            case_elements.append(Paragraph(" | ".join(metrics_str), self.styles['Normal']))
            case_elements.append(Spacer(1, 8))
            
            # Divider
            case_elements.append(Paragraph("_" * 60, self.styles['Normal']))
            case_elements.append(Spacer(1, 12))
            
            elements.append(KeepTogether(case_elements))

        return elements

    def generate(self):
        """Main generation flow."""
        logger.info(f"Generating PDF report at {self.output_path}")
        data = self.load_data()
        
        doc = SimpleDocTemplate(self.output_path, pagesize=letter)
        elements = []

        # Title Page
        elements.append(Paragraph("RAG Evaluation Report", self.styles['Title']))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['Normal']))
        elements.append(Spacer(1, 24))

        # Executive Summary
        winner_name = self._identify_winner(data) or "N/A"
        summary_text = (
            f"This report compares {len(data)} RAG techniques. "
            f"The top performing technique based on overall metrics is <b>{winner_name.replace('_', ' ').title()}</b>."
        )
        elements.append(Paragraph("Executive Summary", self.styles['Heading2']))
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 12))

        # Comparative Table
        elements.append(self._create_summary_table(data))
        elements.append(Spacer(1, 24))

        # Detailed Winner Analysis
        if winner_name and winner_name != "N/A":
            elements.extend(self._create_winner_detail_section(winner_name, data[winner_name]))

        # Build
        doc.build(elements)
        logger.info("PDF Generation Complete")


if __name__ == "__main__":
    json_input_filepath = r"C:\Users\TempAccess\Documents\Dhruv\RAG\17_rag_techniques_output.json"
    pdf_output_filepath = r"C:\Users\TempAccess\Documents\Dhruv\RAG\17_rag_evaluation_report.pdf"
    pdf_report = RAGReportGenerator(json_input_filepath, pdf_output_filepath)
    pdf_report.generate()


    json_input_filepath = r"C:\Users\TempAccess\Documents\Dhruv\RAG\own_rag_techniques_output.json"
    pdf_output_filepath = r"C:\Users\TempAccess\Documents\Dhruv\RAG\own_rag_evaluation_report.pdf"
    pdf_report = RAGReportGenerator(json_input_filepath, pdf_output_filepath)
    pdf_report.generate()

    json_input_filepath = r"C:\Users\TempAccess\Documents\Dhruv\RAG\microsoft_rag_techniques_output.json"
    pdf_output_filepath = r"C:\Users\TempAccess\Documents\Dhruv\RAG\microsoft_rag_evaluation_report.pdf"
    pdf_report = RAGReportGenerator(json_input_filepath, pdf_output_filepath)
    pdf_report.generate()