import os
import sys
from dotenv import load_dotenv

load_dotenv()

from plagiarism_ai.utils.text_chunker import TextChunker
from plagiarism_ai.services.search_service import SearchService

text = 'ஸ்ரீநந்திகியோர் கோயங்காவின் இறுதிச் சடங்குகள் அனைத்தும், கடந்த ஜூலை 15ஆம் தேதி அவர் உருவாக்க உதவிய ஹரியானா ஹிசார் மாவட்டத்தில் இருக்கும் அக்ரோஹா தாமில் உள்ள கோயங்கா உத்யான் பகுதியில் நடைபெற்றது. எஸ்செல் குழுமத் தலைவர் டாக்டர் சுபாஷ் சந்திரா இறுதிச் சடங்குகளை மேற்கொண்டார். பிரதமர் மோடி உள்பட பல்வேறு அரசியல் தலைவர்கள், பிரபலங்கள், நட்சத்திரங்கள், அதிகாரிகள், பொதுமக்கள் என பலரும் ஸ்ரீநந்திகியோர் கோயங்கா மறைவிற்கு ஆழ்ந்த இரங்கலை தெரிவித்தனர்.'

chunks = TextChunker.chunk_text(text)
print('Chunks:', chunks)

search = SearchService()
for chunk in chunks:
    query_exact = f'"{chunk.text}"'
    print('Query (Exact):', query_exact)
    res_exact = search.search(query_exact)
    print('Results Exact:', len(res_exact))
    for r in res_exact:
        has_content = "✅ has content" if r.content else "⚠️  no content (scraper fallback)"
        print(f' - {r.url}  [{has_content}]')
        
    query_broad = chunk.text
    print('Query (Broad):', query_broad)
    res_broad = search.search(query_broad)
    print('Results Broad:', len(res_broad))
    for r in res_broad:
        has_content = "✅ has content" if r.content else "⚠️  no content (scraper fallback)"
        print(f' - {r.url}  [{has_content}]')

