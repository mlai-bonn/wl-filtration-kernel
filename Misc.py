  
'''
Helper method: maps the original graph attribute vectors to simple integers
'''
def simplify_node_labels(g_list):
    for g in g_list:
        for n in g.nodes(data=True):
            if 'label' in n[1]:
                lbl = n[1]['label']
                n[1]['label'] =  int(lbl[0])
            else:
                n[1]['label'] = 0