"""Loss masks over original response text. Never reserialize the teacher answer."""
from .data import object_spans

def render(tokenizer,prompt,answer):
    marker='CAM_LDS_BOUNDARY_f91b7';user_marker='CAM_LDS_USER_f91b7'
    probe=tokenizer.apply_chat_template([{'role':'user','content':user_marker},
        {'role':'assistant','content':marker}],tokenize=False)
    if probe.count(marker)!=1 or probe.count(user_marker)!=1: raise ValueError('Ambiguous chat boundary')
    prefix,ending=probe.split(marker)
    if user_marker not in prefix:raise ValueError('User content not before assistant')
    # Native role/control-token wrapper; insert content after rendering so Jinja
    # trim filters cannot silently modify source whitespace in either message.
    prefix=prefix.replace(user_marker,prompt)
    full=prefix+answer+ending
    if not ending:
        ending=tokenizer.eos_token
        if not ending: raise ValueError('No EOS marker')
        full+=ending
    return prefix,full,ending

def encode_exact(tokenizer,row):
    answer=row['teacher_response_raw']; prefix,full,ending=render(tokenizer,row['prompt_text'],answer)
    encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    try: spans=object_spans(answer)
    except (ValueError,IndexError): spans={}
    known={'techniques':2,'confidence':3,'explanation':4}
    b=len(prefix); e=b+len(answer); fields=[]; full_labels=[]
    for (start,stop),tid in zip(encoded['offset_mapping'],encoded['input_ids']):
        full_labels.append(tid if start>=b and stop>start else -100)
        if start<b or start==stop: fields.append(0); continue
        if start>=e: fields.append(1); continue
        hits=[(left,right,known[key]) for key,(left,right) in spans.items()
              if key in known and start<b+right and stop>b+left]
        if not hits: fields.append(1 if spans else 0)
        elif len(hits)==1 and start>=b+hits[0][0] and stop<=b+hits[0][1]: fields.append(hits[0][2])
        else: fields.append(0)
    ending_ids=tokenizer(ending,add_special_tokens=False)['input_ids']
    if not ending_ids or encoded['input_ids'][-len(ending_ids):]!=ending_ids:
        raise ValueError('Chat ending cannot be identified safely')
    for i in range(len(fields)-len(ending_ids),len(fields)):
        fields[i]=1; full_labels[i]=encoded['input_ids'][i]
    return {'input_ids':encoded['input_ids'],'fields':fields,'labels':full_labels,
            'prompt_id':row['prompt_id']}
