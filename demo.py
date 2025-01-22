import gradio as gr


def on_button_down(cart: gr.State):
    return len(cart)


with gr.Blocks() as demo:
    cart = gr.State([])
    items_to_add = gr.CheckboxGroup(["Cereal", "Milk", "Orange Juice", "Water"])


    def add_items(new_items, previous_cart):
        cart = previous_cart + new_items
        return cart


    gr.Button("Add Items").click(add_items, [items_to_add, cart], cart)

    cart_size = gr.Number(label="Cart Size")
    gr.Button("Get Cart Size").click(fn=on_button_down, inputs=cart, outputs=cart_size)

demo.launch()
