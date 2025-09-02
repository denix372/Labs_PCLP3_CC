import gradio as gr

def regresor(x):
    return 2 * x + 1

interfata = gr.Interface(
    fn=regresor,
    inputs=gr.Number(label="x"),
    outputs=gr.Number(label="y prezis")
)

interfata.launch()