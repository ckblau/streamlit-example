import streamlit as st

from user import use_model

num_attr = []
norm_attr = []
num_input = dict()
norm_input = dict()


def parse_attr(text, norm=False):
    lines = text.split("\n")
    attr = []
    cur = []
    for line in lines:
        if line == "":
            attr += [cur]
            cur = []
        else:
            if norm:
                cur += [line.split("#")]
            else:
                cur += [line]
    return attr


def load_model():
    global num_attr, norm_attr, num_input, norm_input

    f = open("num.txt", "r")
    num_attr = parse_attr(f.read())
    f = open("norm.txt", "r")
    norm_attr = parse_attr(f.read(), norm=True)
    # num_attr_size = len(num_attr)
    # norm_attr_size = len(norm_attr)

    for i, attr in enumerate(num_attr):
        a_label = num_attr[i][0]
        a_help = num_attr[i][1]
        a_key = "num_attr_{}".format(i)
        cur = st.number_input(a_label, help=a_help, key=a_key, value=-1)
        num_input[a_label] = cur

    for i, attr in enumerate(norm_attr):
        a_label = norm_attr[i][0][0]
        a_help = norm_attr[i][1][0]
        a_key = "norm_attr_{}".format(i)
        a_options = [""] + [e[1] for e in norm_attr[i][2:]]
        cur = st.selectbox(a_label, help=a_help, options=a_options, index=0, key=a_key)
        cur_real = [e[0] for e in norm_attr[i][2:] if cur == e[1]]
        if cur_real:
            norm_input[a_label] = cur_real[0]


def submit_result():
    print(num_input)
    print(norm_input)
    total_input = num_input | norm_input
    if "MS SubClass" not in total_input:
        total_input["MS SubClass"] = -1
    # st.write(total_input)
    pred = use_model(total_input)
    # st.write(pred)
    return pred


def main():
    st.title("House Price Prediction")
    load_state = st.text("Loading model...")
    load_model()
    load_state.text("Waiting for input...")
    submit_state = st.text("")
    if st.button("Submit"):
        submit_state.text("Submitted!")
        pred = submit_result()
        st.text("Price predicted: ${}".format(pred[0]))
    else:
        submit_state.text("Press button to submit...")


if __name__ == '__main__':
    main()
