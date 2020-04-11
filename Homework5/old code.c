/*BlockIndices* GetBlockIndices(size_t n, size_t block_dim)
{
    size_t num_blocks = block_dim * block_dim;
#ifdef GPU
    BlockIndices* ret;
    cudaMallocManaged(&ret, num_blocks*sizeof(BlockIndices));
#else
    BlockIndices* ret = new BlockIndices[num_blocks];
#endif
    size_t block_size = n / block_dim;

    size_t x_min, x_max, y_min, y_max;
    size_t block_index = 0;
    for(size_t i = 0; i < block_dim; i++)
    {
        x_min = i * block_size;
        x_max = (i+1) * block_size;
        for(size_t j = 0; j < block_dim; j++)
        {
            y_min = j * block_size;
            y_max = (j+1) * block_size;

            ret[block_index].x_min = x_min;
            ret[block_index].x_max = x_max;
            ret[block_index].y_min = y_min;
            ret[block_index].y_max = y_max;
            block_index++;
        }
        
    }
    return ret;
}*/
