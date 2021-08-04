
vowel = {'a', 'e', 'i', 'o', 'u', 'y'}
def vowel_adj(input):
    prev = None
    returned = False
    for i in input:
        if prev == None:
            prev = i
            continue
        returned2 = False
        if prev in vowel:
            print(i)
            returned2 = True
        if i in vowel and not returned:
            print (prev)
        prev = i
        returned = returned2


if __name__ == '__main__':
    # vowel_adj('')
    # vowel_adj('123')
    vowel_adj('aaaa')
    #vowel_adj('example')