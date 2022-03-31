from utils import *
def get_graph(a,b,c,data):

    def load_kg(dataset=TMP_DIR[data]):
        kg_file = dataset + '/kg.pkl'
        kg = pickle.load(open(kg_file, 'rb'))
        return kg
    kg=load_kg()
    from tqdm import tqdm
    u_ui=[]
    v_ui=[]
    for name in kg.G.keys():
        for n in tqdm(kg.G[name]):
            for item in kg.G[name][n]['interact']:
                u_ui.append(int(n))
                v_ui.append(int(item))
        break
    u_uu=[]
    v_uu=[]
    for n in tqdm(kg.G['user']):
        for item in kg.G['user'][n]['friends']:
            u_uu.append(int(n))
            v_uu.append(int(item))
    u_ua=[]
    v_ua=[]
    for n in tqdm(kg.G['user']):
        for item in kg.G['user'][n]['like']:
            u_ua.append(int(n))
            v_ua.append(int(item))
    u_ia=[]
    v_ia=[]
    for n in tqdm(kg.G['item']):
        for item in kg.G['item'][n]['belong_to']:
            u_ia.append(int(n))
            v_ia.append(int(item))

    G = dgl.heterograph({
        ('user', 'friends', 'user'): (u_uu, v_uu),
        ('user', 'friends', 'user'): (v_uu,u_uu),
        ('user', 'interact', 'item'): (u_ui, v_ui),
        ('item', 'interact', 'user'): (v_ui,u_ui),
        ('user', 'like', 'attribute'): (u_ua, v_ua),
        ('attribute', 'like', 'user'): (v_ua,u_ua),
        ('item', 'belong_to', 'attribute'): (u_ia, v_ia),
        ('attribute', 'belong_to', 'item'): (v_ia,u_ia),
        
    },num_nodes_dict={'item':b,'user':a,'attribute':c+1})
    print("item:{},user:{},feature:{}".format(b,a,c))
    return G