import gradio as gr

def clasificator(x1, x2):
    return "Clasa A" if x1 + x2 > 1 else "Clasa B"

interfata = gr.Interface(
    fn=clasificator,
    inputs=[gr.Number(label="x1"), gr.Number(label="x2")],
    outputs="text"
)

interfata.launch()