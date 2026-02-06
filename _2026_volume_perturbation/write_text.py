def write_text(df, out_path):
    df.write_csv(out_path, separator=" ", include_header=False, quote_style="never")