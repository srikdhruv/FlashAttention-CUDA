def max_block_size(M, N, blocksize=32):
    num_blocks_y = int((M-1+blocksize)/blocksize)
    num_blocks_x = int((N-1+blocksize)/blocksize)
    total_blocks = num_blocks_y*num_blocks_x
    print(total_blocks, total_blocks < 2147483647)


max_block_size(128, 128, 32)