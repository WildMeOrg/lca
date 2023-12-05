
import numpy as np


def intersect_stats(df_a, df_b, df_c=None, key="name", a_name="train", b_name="test", c_name="val"):
    print("** cross-set stats **")
    print()
    print('- Counts: ')
    names_a = df_a[key].unique()
    names_b = df_b[key].unique()


    print(f"number of individuals in {a_name}: ", len(names_a))
    print(f"number of annotations in {a_name}: ", len(df_a))
    print()
    print(f"number of individuals in {b_name}: ", len(names_b))
    print(f"number of annotations in {b_name}: ", len(df_b))
    print()
    if df_c is not None:
        names_c = df_c[key].unique()
        print(f"number of individuals in {c_name}: ", len(names_c))
        print(f"number of annotations in {c_name}: ", len(df_c))
        print()
    print(f"average number of annotations per individual in {a_name}: {len(df_a) / len(names_a):.2f}")
    print(f"average number of annotations per individual in {b_name}: {len(df_b) / len(names_b):.2f}")
    if df_c is not None:
        print(f"average number of annotations per individual in {c_name}: {len(df_c) / len(names_c):.2f}")
        print()

    print('- New individuals: ')
    names_diff_ab = np.setdiff1d(names_b, names_a)
    
    print(f"number of new (unseen) individuals in {b_name}: {len(names_diff_ab)}")
    print(f"ratio of new names to all individuals in {b_name}: {len(names_diff_ab) / len(names_b):.2f}")
    print()
    if df_c is not None:
        names_diff_ac = np.setdiff1d(names_c, names_a)
        print(f"number of new (unseen) individuals in {c_name}: {len(names_diff_ac)}")
        print(f"ratio of new names to all individuals in {c_name}: {len(names_diff_ac) / len(names_c):.2f}")

    print("- Individuals in sets: ")
    len_intersect_ab = len(np.intersect1d(names_a, names_b))
    
    print(f"number of overlapping individuals in {a_name} & {b_name}: {len_intersect_ab}")
    print(f"ratio of overlapping names to total individuals in {a_name}: {len_intersect_ab / len(names_a):.2f}")
    print(f"ratio of overlapping names to total individuals in {b_name}: {len_intersect_ab / len(names_b):.2f}")

    if df_c is not None:
        len_intersect_ac = len(np.intersect1d(names_a, names_c))
        print(f"number of overlapping individuals in {a_name} & {c_name}: {len_intersect_ac}")
        print(f"ratio of overlapping names to total individuals in {a_name}: {len_intersect_ac / len(names_a):.2f}")
        print(f"ratio of overlapping names to total individuals in {c_name}: {len_intersect_ac / len(names_c):.2f}")


def get_basic_stats(df_stat, min_filt=3, max_filt=None, individual_key='name'):

    if min_filt:
        df_stat = df_stat.groupby(individual_key).filter(lambda g: len(g)>=min_filt)
        print(f'Min filtering applied: {min_filt}')
    if max_filt:
        df_stat = df_stat.groupby(individual_key).head(10)
        print(f'Max subsampling applied: {max_filt}')
    avg = (len(df_stat) / df_stat[individual_key].nunique() )

    print('Number of individuals:', len(df_stat[individual_key].unique()))
    print('Number of annotations:', len(df_stat))
    
    print(f'Average number of images per individual: {avg:.2f}')


def do_split_summary(df1, df2=None, df3=None):
    print('\n ** Species value counts ** \n')
    print(df1['species'].value_counts())

    print('\n** Basic dataset stats **\n')
    get_basic_stats(df1)

    print()
    print(df1)
    print(df2)
    print(df3)
    if df2 is not None:
        intersect_stats(df1, df2, df3, key="individual_id")

    df1['species'].value_counts().plot(kind='barh')