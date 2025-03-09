from googletrans import Translator

translator = Translator()

def translate_text(original_text: str):
    if not isinstance(original_text, str) or not original_text.strip():
        return "", ""  # Return empty strings if input is not valid
    
    translation = translator.translate(original_text, dest='en')
    return translation.text, translation.src
