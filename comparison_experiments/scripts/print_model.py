from graphviz import Digraph

def parse_model_structure(model_text: str):
    lines = model_text.strip().split('\n')
    graph = Digraph(format='png')
    stack = []
    node_id = 0

    def new_node(name, label):
        nonlocal node_id
        nid = f"node{node_id}"
        graph.node(nid, label)
        node_id += 1
        return nid

    last_indent = 0
    parent_stack = []

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if not stripped or stripped.startswith('('):  # skip lines like '(' or empty
            continue

        name = stripped.split(':')[0].strip()
        label = stripped.strip()

        this_node = new_node(name, label)

        if not parent_stack:
            parent_stack.append((indent, this_node))
        else:
            while parent_stack and parent_stack[-1][0] >= indent:
                parent_stack.pop()

            if parent_stack:
                graph.edge(parent_stack[-1][1], this_node)
            parent_stack.append((indent, this_node))

    return graph


# 示例：读取你打印好的模型结构
with open("model.txt", "r") as f:
    model_str = f.read()

# 生成图
graph = parse_model_structure(model_str)
graph.render("unet_structure", view=True)  # 会保存 unet_structure.png 并打开
