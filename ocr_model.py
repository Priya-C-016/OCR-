from PIL import Image
from PIL import ImageEnhance
from spellchecker import SpellChecker
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def extract_text(image_path):
    spell = SpellChecker()
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    #image = Image.open(image_path).convert("RGB")
    image = Image.open(image_path).convert("RGB") 
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    corrected_words = [spell.correction(word) for word in extracted_text.split()]
    corrected_text = " ".join(corrected_words)

    return corrected_text
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  
    corrected_text = extract_text(image_path)
    print("Corrected Text:", corrected_text)
