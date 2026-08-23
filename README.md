# sfematch
Subject-Field-Expert matching

# Data collection 

Start by compiling a list of experts in a text file (each in a line: firstname; family name) like in [](./examples/info/expert_list.txt).
Execute this command to get a file like this [](./examples/info/experts_info_choices.json)

```
>> python exec/collect_infos.py examples/info/expert_list.txt  --out examples/info/experts_info_choices.json
```

Then manually choose the right IDs and compile a file similar to [](./examples/info/experts_info.json).

