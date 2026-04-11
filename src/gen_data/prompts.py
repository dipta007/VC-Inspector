PROMPT_SET = dict(
    obj_list="""\
### Instruction:
Given the input text, generate a list of objects in the caption in the format of ["Object1", "Object2", ...]. Don't include any verbs. ONLY REPLY THE ANSWER.

### Input: {cap}

### Output:
""",
    act_list="""\
### Instruction:
Given the input text, generate a list of actions in the caption in the format of ["Action1", "Action2", ...]. ONLY REPLY THE ANSWER.

### Input: {cap}

### Output:
""",
    new_obj="""\
### Instruction:
Find the parent class of the given object and generate one of its child class that has a different meaning but shares the same parent. The new class cannot be a synonym or similar terms to the original object. It can be an antonym or any co-hyponym. For example, generate "dog" for "cat". ONLY REPLY THE NEW CLASS.

### Input: {obj}

### Output:
""",
    new_act="""\
### Instruction:
Find a different action that the subject can perform which has different meaning than the input action. The new action cannot be a synonym or similar terms to the original action. For example, generate "put into" for "take out of". ONLY REPLY THE NEW ACTION.

### Input: {act}

### Output:
""",
    substitute="""\
### Instruction:
Substitute "{ow}" in "{cap}" as "{nw}". Keep the answer in the same format as {cap}. ONLY REPLY THE ANSWER.

### Input: {cap}

### Output:
""",
)
