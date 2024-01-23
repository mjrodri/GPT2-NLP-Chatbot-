import tkinter as tk
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Monkey-patch the deprecated function
tf.losses.sparse_softmax_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Chatbot GUI")

        self.model, self.tokenizer = self.initialize_model_and_tokenizer()

        self.label = tk.Label(master, text="You:")
        self.label.pack()

        self.user_input = tk.Entry(master)
        self.user_input.pack()

        self.chatbox = tk.Text(master, height=10, width=40)
        self.chatbox.pack()

        self.exit_button = tk.Button(master, text="Exit", command=self.master.destroy)
        self.exit_button.pack()

        self.chat_button = tk.Button(master, text="Chat", command=self.chat)
        self.chat_button.pack()

    def initialize_model_and_tokenizer(self):
        model_name = "gpt2"
        model = TFGPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_response(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="tf")
        output_ids = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def chat(self):
        user_input = self.user_input.get()
        if user_input.lower() == 'exit':
            self.master.destroy()
            return

        response = self.generate_response(user_input)
        self.chatbox.insert(tk.END, "You: " + user_input + "\n")
        self.chatbox.insert(tk.END, "ChatGPT: " + response + "\n")
        self.user_input.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()