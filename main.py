import tkinter as tk
from tkinter import filedialog, Label, Button, BOTTOM, LEFT, RIGHT, Frame, END, Scale, HORIZONTAL, messagebox
from PIL import ImageTk, Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np
import cv2
import os
from transformers import pipeline
import torch
import sys



class CaptionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry('1400x900')  # Further increased window size for better visibility
        self.root.title('Advanced Image Captioning App')
        self.root.configure(background='#1F1F1F')  # Darker background for aesthetics

        # Initialize current caption
        self.current_caption = ''

        # Initialize Hugging Face pipeline
        self.initialize_pipeline()

        # Main Frame
        self.main_frame = Frame(root, bg='#1F1F1F')
        self.main_frame.pack(fill='both', expand=True)

        # Heading
        self.heading = Label(
            self.main_frame,
            text="Advanced Image Captioning App",
            pady=20,
            font=('Helvetica', 32, 'bold'),  # Increased font size
            bg='#1F1F1F',
            fg='#FFFFFF'
        )
        self.heading.pack()

        # Image Display Frame
        self.image_frame = Frame(self.main_frame, bg='#1F1F1F')
        self.image_frame.pack(pady=20)

        self.sign_image = Label(self.image_frame, bg='#1F1F1F')
        self.sign_image.pack()

        # Caption Display
        self.label = Label(
            self.main_frame,
            background='#1F1F1F',
            font=('Helvetica', 20),  # Increased font size
            fg='#00FF00',  # Bright green for visibility
            wraplength=1200,
            justify='center'
        )
        self.label.pack(pady=20)

        # Button Frame
        self.button_frame = Frame(self.main_frame, bg='#1F1F1F')
        self.button_frame.pack(pady=10)

        # Load Image Button
        self.upload_button = Button(
            self.button_frame,
            text="Load Image",
            command=self.upload_image,
            padx=30,
            pady=15,
            bg='#2980B9',
            fg='white',
            font=('Helvetica', 14, 'bold'),  # Increased font size
            borderwidth=0,
            activebackground='#1ABC9C'
        )
        self.upload_button.grid(row=0, column=0, padx=20, pady=10)

        # Generate Caption Button
        self.generate_button = Button(
            self.button_frame,
            text="Generate Caption",
            command=self.generate_caption,
            padx=30,
            pady=15,
            bg='#27AE60',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            borderwidth=0,
            activebackground='#2ECC71'
        )
        self.generate_button.grid(row=0, column=1, padx=20, pady=10)

        # Save Image Button
        self.save_button = Button(
            self.button_frame,
            text="Save Image with Caption",
            command=self.save_captioned_image,
            padx=30,
            pady=15,
            bg='#C0392B',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            borderwidth=0,
            activebackground='#E74C3C'
        )
        self.save_button.grid(row=0, column=2, padx=20, pady=10)

        # Copy Caption Button
        self.copy_button = Button(
            self.button_frame,
            text="Copy Caption",
            command=self.copy_to_clipboard,
            padx=30,
            pady=15,
            bg='#8E44AD',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            borderwidth=0,
            activebackground='#9B59B6'
        )
        self.copy_button.grid(row=0, column=3, padx=20, pady=10)

        # OpenCV Functionality Buttons Frame
        self.cv_button_frame = Frame(self.main_frame, bg='#1F1F1F')
        self.cv_button_frame.pack(pady=20)

        # Grayscale Button
        self.gray_button = Button(
            self.cv_button_frame,
            text="Grayscale",
            command=self.apply_grayscale,
            padx=25,
            pady=12,
            bg='#34495E',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            borderwidth=0,
            activebackground='#2C3E50'
        )
        self.gray_button.grid(row=0, column=0, padx=10, pady=10)

        # Edge Detection Button
        self.edge_button = Button(
            self.cv_button_frame,
            text="Edge Detection",
            command=self.apply_edge_detection,
            padx=25,
            pady=12,
            bg='#34495E',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            borderwidth=0,
            activebackground='#2C3E50'
        )
        self.edge_button.grid(row=0, column=1, padx=10, pady=10)

        # Brightness Adjustment
        self.brightness_button = Button(
            self.cv_button_frame,
            text="Adjust Brightness",
            command=self.adjust_brightness,
            padx=25,
            pady=12,
            bg='#34495E',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            borderwidth=0,
            activebackground='#2C3E50'
        )
        self.brightness_button.grid(row=0, column=2, padx=10, pady=10)

        # Contrast Adjustment
        self.contrast_button = Button(
            self.cv_button_frame,
            text="Adjust Contrast",
            command=self.adjust_contrast,
            padx=25,
            pady=12,
            bg='#34495E',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            borderwidth=0,
            activebackground='#2C3E50'
        )
        self.contrast_button.grid(row=0, column=3, padx=10, pady=10)

        # Blur Image
        self.blur_button = Button(
            self.cv_button_frame,
            text="Blur",
            command=self.apply_blur,
            padx=35,
            pady=12,
            bg='#34495E',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            borderwidth=0,
            activebackground='#2C3E50'
        )
        self.blur_button.grid(row=0, column=4, padx=10, pady=10)

        # Reset Image Button
        self.reset_button = Button(
            self.cv_button_frame,
            text="Reset Image",
            command=self.reset_image,
            padx=25,
            pady=12,
            bg='#E74C3C',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            borderwidth=0,
            activebackground='#C0392B'
        )
        self.reset_button.grid(row=0, column=5, padx=10, pady=10)

        # To store the original and processed images
        self.original_image = None
        self.processed_image = None

    def initialize_pipeline(self):
        try:
            self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
            self.pipe = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-large",
                device=self.device
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model pipeline: {e}")
            self.root.destroy()

    def preprocess_image_for_captioning(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess image: {e}")
            return None

    def generate_caption(self):
        if not self.processed_image:
            self.label.configure(foreground='#E74C3C', text='Please load an image first.')
            return

        try:
            pil_image = self.preprocess_image_for_captioning(self.processed_image)
            if pil_image is None:
                return

            # Generate caption using the pipeline
            output = self.pipe(pil_image)

            if isinstance(output, list) and len(output) > 0:
                self.current_caption = output[0]['generated_text']
            else:
                self.current_caption = "No caption generated."

            self.label.configure(foreground='#00FF00', text='Caption: ' + self.current_caption)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate caption: {e}")

    def save_captioned_image(self):
        if not self.processed_image or not self.current_caption:
            self.label.configure(foreground='#E74C3C', text='No image or caption to save.')
            return

        try:
            # Load the image using OpenCV
            image = cv2.imread(self.processed_image)
            if image is None:
                self.label.configure(foreground='#E74C3C', text='Error loading image for saving.')
                return

            # Define the position for the caption
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Increased font scale for better visibility
            thickness = 1
            color = (255, 255, 255)  # White color

            # Calculate text size to adjust placement
            text_size, _ = cv2.getTextSize(self.current_caption, font, font_scale, thickness)
            text_width, text_height = text_size

            # Position the text at the top-center of the image
            position = ((image.shape[1] - text_width) // 2, text_height + 30)

            # Add rectangle for better visibility of text
            cv2.rectangle(
                image,
                (position[0] - 10, position[1] - text_height - 10),
                (position[0] + text_width + 10, position[1] + 10),
                (0, 0, 0),
                -1
            )

            # Put the caption text on the image
            cv2.putText(image, self.current_caption, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Save the image
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if save_path:
                cv2.imwrite(save_path, image)
                self.label.configure(foreground='#2ECC71', text='Image saved with caption.')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def copy_to_clipboard(self):
        caption = self.current_caption
        if caption:
            self.root.clipboard_clear()
            self.root.clipboard_append(caption)
            self.label.configure(foreground='#2ECC71', text='Caption copied to clipboard.')
        else:
            self.label.configure(foreground='#E74C3C', text='No caption to copy.')

    def show_classify_button(self, file_path):
        pass  # Buttons are always visible

    def clear_image(self):
        self.sign_image.configure(image='')
        self.label.configure(text='')
        self.original_image = None
        self.processed_image = None

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            )
            if file_path:
                self.original_image = file_path
                self.processed_image = file_path
                uploaded = Image.open(file_path)
                uploaded.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
                im = ImageTk.PhotoImage(uploaded)
                self.sign_image.configure(image=im)
                self.sign_image.image = im
                self.label.configure(text='')
                self.current_caption = ''
        except Exception as e:
            messagebox.showerror("Error", f"Error uploading image: {e}")

    def apply_grayscale(self):
        if self.processed_image:
            try:
                image = cv2.imread(self.processed_image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('temp_gray.png', gray)
                im = Image.open('temp_gray.png').convert("RGB")
                im.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
                im = ImageTk.PhotoImage(im)
                self.sign_image.configure(image=im)
                self.sign_image.image = im
                self.processed_image = 'temp_gray.png'
            except Exception as e:
                messagebox.showerror("Error", f"Error applying grayscale: {e}")

    def apply_edge_detection(self):
        if self.processed_image:
            try:
                image = cv2.imread(self.processed_image)
                edges = cv2.Canny(image, 100, 200)
                cv2.imwrite('temp_edges.png', edges)
                im = Image.open('temp_edges.png').convert("RGB")
                im.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
                im = ImageTk.PhotoImage(im)
                self.sign_image.configure(image=im)
                self.sign_image.image = im
                self.processed_image = 'temp_edges.png'
            except Exception as e:
                messagebox.showerror("Error", f"Error applying edge detection: {e}")

    def adjust_brightness(self):
        if self.processed_image:
            self.brightness_window = tk.Toplevel(self.root)
            self.brightness_window.title("Adjust Brightness")
            self.brightness_window.geometry("500x150")
            self.brightness_window.configure(bg='#1F1F1F')

            tk.Label(
                self.brightness_window,
                text="Brightness",
                fg='white',
                bg='#1F1F1F',
                font=('Helvetica', 14)
            ).pack(pady=20)
            self.brightness_scale = Scale(
                self.brightness_window,
                from_=0,
                to=100,
                orient=HORIZONTAL,
                command=self.apply_brightness_change,
                length=400
            )
            self.brightness_scale.set(50)
            self.brightness_scale.pack()
        else:
            self.label.configure(foreground='#E74C3C', text='Please load an image first.')

    def apply_brightness_change(self, val):
        try:
            value = int(val)
            image = cv2.imread(self.processed_image)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (value / 50)  # 50 is the midpoint
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite('temp_bright.png', bright)
            im = Image.open('temp_bright.png').convert("RGB")
            im.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
            im = ImageTk.PhotoImage(im)
            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.processed_image = 'temp_bright.png'
        except Exception as e:
            messagebox.showerror("Error", f"Error adjusting brightness: {e}")

    def adjust_contrast(self):
        if self.processed_image:
            self.contrast_window = tk.Toplevel(self.root)
            self.contrast_window.title("Adjust Contrast")
            self.contrast_window.geometry("500x150")
            self.contrast_window.configure(bg='#1F1F1F')

            tk.Label(
                self.contrast_window,
                text="Contrast",
                fg='white',
                bg='#1F1F1F',
                font=('Helvetica', 14)
            ).pack(pady=20)
            self.contrast_scale = Scale(
                self.contrast_window,
                from_=0,
                to=100,
                orient=HORIZONTAL,
                command=self.apply_contrast_change,
                length=400
            )
            self.contrast_scale.set(50)
            self.contrast_scale.pack()
        else:
            self.label.configure(foreground='#E74C3C', text='Please load an image first.')

    def apply_contrast_change(self, val):
        try:
            value = int(val)
            image = cv2.imread(self.processed_image)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab = np.array(lab, dtype=np.float64)
            lab[:, :, 0] = lab[:, :, 0] * (value / 50)  # 50 is the midpoint
            lab[:, :, 0][lab[:, :, 0] > 255] = 255
            lab = np.array(lab, dtype=np.uint8)
            contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite('temp_contrast.png', contrast)
            im = Image.open('temp_contrast.png').convert("RGB")
            im.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
            im = ImageTk.PhotoImage(im)
            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.processed_image = 'temp_contrast.png'
        except Exception as e:
            messagebox.showerror("Error", f"Error adjusting contrast: {e}")

    def apply_blur(self):
        if self.processed_image:
            try:
                image = cv2.imread(self.processed_image)
                blurred = cv2.GaussianBlur(image, (15, 15), 0)
                cv2.imwrite('temp_blur.png', blurred)
                im = Image.open('temp_blur.png').convert("RGB")
                im.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
                im = ImageTk.PhotoImage(im)
                self.sign_image.configure(image=im)
                self.sign_image.image = im
                self.processed_image = 'temp_blur.png'
            except Exception as e:
                messagebox.showerror("Error", f"Error applying blur: {e}")
        else:
            self.label.configure(foreground='#E74C3C', text='Please load an image first.')

    def reset_image(self):
        if self.original_image:
            try:
                uploaded = Image.open(self.original_image)
                uploaded.thumbnail(((self.main_frame.winfo_width() / 2.5), (self.main_frame.winfo_height() / 2.5)))
                im = ImageTk.PhotoImage(uploaded)
                self.sign_image.configure(image=im)
                self.sign_image.image = im
                self.processed_image = self.original_image
                self.label.configure(text='')
                self.current_caption = ''
            except Exception as e:
                messagebox.showerror("Error", f"Error resetting image: {e}")


def main():
    root = tk.Tk()
    app = CaptionGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()