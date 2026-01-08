

def zero_shot_prompt(text, labels):
    return f"""<s>[INST]
    You are a medical domain classification assistant.

    Your task is to classify the medical text into exactly ONE medical specialty
    from the list below.

    {", ".join(labels)}

    IMPORTANT RULES:
    - Choose the SINGLE most appropriate specialty
    - If multiple specialties apply, choose the PRIMARY one
    - Do NOT list multiple labels
    - Do NOT number labels
    - Output the label only, exactly as written

    Medical text:
    "{text}"
    [/INST]
    """

def five_shot_prompt(text, labels):
    return f"""<s>[INST]
    You are a medical domain classification assistant.

    Your task is to classify the medical text into exactly ONE medical specialty
    from the list below.

    {", ".join(labels)}
    IMPORTANT RULES:
    - Choose the SINGLE most appropriate specialty
    - If multiple specialties apply, choose the PRIMARY one
    - Do NOT list multiple labels
    - Do NOT number labels
    - Output the label only, exactly as written

    Examples:

    Medical text:
    "PREOPERATIVE DIAGNOSIS:,  Nuclear sclerotic cataract, right eye.OPERATIVE PROCEDURES:,  Kelman phacoemulsification with posterior chamber intraocular lens, right eye.This is a 40-year-old male, who has been noticing problems with blurry vision." 
    Label:
    Ophthalmology

    Medical text:
    "IDENTIFICATION OF PATIENT: , This is a 31-year-old female who was referred by herself.  She was formerly seen at Counseling Center.  She is a reliable historian.,CHIEF COMPLAINT:, I'm bipolar and I have severe anxiety disorder.  I have posttraumatic stress syndrome."
    Label:
    Psychiatry / Psychology

    Medical text:
    "PREOPERATIVE DIAGNOSES:,1.  Bunion left foot.,2.  Hammertoe, left second toe.,POSTOPERATIVE DIAGNOSES:,1.  Bunion left foot.,2.  Hammertoe, left second toe.,PROCEDURE PERFORMED:,1.  Bunionectomy, SCARF type, with metatarsal osteotomy and internal screw fixation, left.,2.  Arthroplasty left second toe."
    Label:
    Podiatry

    Medical text:
    "HISTORY: , The patient is a 15-year-old female who was seen in consultation at the request of Dr. X on 05/15/2008 regarding enlarged tonsils.  The patient has been having difficult time with having two to three bouts of tonsillitis this year.  She does average about four bouts of tonsillitis per year for the past several years.  She notes that throat pain and fever with the actual infections."
    Label:
    ENT - Otolaryngology

    Medical text:
    "The final histologic diagnosis of the tissue removed during that procedure revealed focal fibrosis.  No calcifications could be identified in examination of the biopsy material including radiograph taken of the preserved tissue.,Two days post stereotactic core needle biopsy, however, the patient returned to the breast center with severe swelling and pain and mass in the left breast."
    Label:
    Obstetrics / Gynecology

    Now classify the following text.

    Medical text:
    "{text}"
    [/INST]
    """
