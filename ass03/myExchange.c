
static inline void exchangeGhostRows(int myrank, int numranks, size_t realHeight, size_t totalWidth) {
    MPI_Request send_requests[2], recv_requests[2];
    int prev_rank = (myrank - 1 + numranks) % numranks;
    int next_rank = (myrank + 1) % numranks;

    // Correct exchange logic
    MPI_Isend(g_data + totalWidth, totalWidth, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, &send_requests[0]); // Send top real row to prev
    MPI_Isend(g_data + totalWidth * (realHeight - 2), totalWidth, MPI_UNSIGNED_CHAR, next_rank, 1, MPI_COMM_WORLD, &send_requests[1]); // Send bottom real row to next

    MPI_Irecv(g_data, totalWidth, MPI_UNSIGNED_CHAR, prev_rank, 1, MPI_COMM_WORLD, &recv_requests[0]); // Receive top ghost row from prev
    MPI_Irecv(g_data + totalWidth * (realHeight - 1), totalWidth, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &recv_requests[1]); // Receive bottom ghost row from next

    MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(2, send_requests, MPI_STATUSES_IGNORE);
}