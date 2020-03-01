#include <unistd.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

#define RUN_MULTITHREADED 1
#define NUM_THREADS 6

#define MIN(X, Y) X < Y ? X : Y
#define MAX(X, Y) X < Y ? Y : X

typedef int bool;
#define false 0
#define true 1

struct timespec start_time;
struct timespec stop_time;

typedef struct game {
	bool* cells;
	bool* next_cells;
	int size;
} game;

typedef struct game_board_section {
	int start_row;
	int end_row;
	int start_col; 
	int end_col;
} game_board_section;

typedef struct thread_context {
	game_board_section* section;
	sem_t board_semaphore;
	sem_t result_semaphore;
	bool section_dead;
} thread_context;

game game_board;

void start_clock()
{
	clock_gettime(CLOCK_REALTIME, &start_time);
}
void stop_clock()
{
	clock_gettime(CLOCK_REALTIME, &stop_time);
}
double get_clock_result_seconds()
{
	double result = stop_time.tv_sec - start_time.tv_sec;
	result += (double)(stop_time.tv_nsec - start_time.tv_nsec) / 1000000000;
	return result;
}
void print_time_seconds(double seconds)
{
	printf("%0.9f seconds", seconds);
}
bool* GetCell(int row, int col, bool* cells)
{
	//store/access in row major order
	return &cells[row * game_board.size + col];
} 

int GetNumLivingNeighbors(int column, int row)
{
	int row_start = MAX(row - 1, 0);
	int row_end = MIN(row + 1, game_board.size - 1);
	int col_start = MAX(column - 1, 0);
	int col_end = MIN(column + 1, game_board.size - 1);
	int sum = 0;
	for(int i = col_start; i <= col_end; i++)
	{
		for(int j = row_start; j <= row_end; j++)
		{
			if (i == column && j == row)
				continue;
			if(*GetCell(j, i, game_board.cells))
				sum++;
		}
	}
	return sum;
}

void GameBoardFlip()
{
	bool* temp = game_board.cells;
	game_board.cells = game_board.next_cells;
	game_board.next_cells = temp;
}
void AllocateGameBoard(int size)
{
	game_board.cells = malloc(sizeof(bool) * size * size);
	game_board.next_cells = malloc(sizeof(bool) * size * size);
	game_board.size = size;
}
void LivingEdgesInitGameBoard(int size)
{
	int num_living_cells = 0;

	//initialize the whole board to false
	memset(game_board.cells, false, game_board.size * game_board.size * sizeof(bool));

	//initialize first and last row 
	for(int col = 0; col < game_board.size; col++)
	{
		*GetCell(0, col, game_board.cells) = true;
		*GetCell(size-1, col, game_board.cells) = true;
		num_living_cells+=2;
	}
	
	//initialize first and last column of each remaining row
	for(int row = 1; row < game_board.size-1; row++)
	{
		*GetCell(row, 0, game_board.cells) = true;
		*GetCell(row, game_board.size-1, game_board.cells) = true;
		num_living_cells+=2;
	}
	

	printf("Of %d total cells, %d are alive.\n", game_board.size * game_board.size, num_living_cells);
}
void PrintGameBoard()
{
#ifdef DISABLE_PRINTING
	return;
#else
	for(int row = 0; row < game_board.size; row++)
	{	
		for(int col = 0; col < game_board.size; col++)
		{
			if(*GetCell(row, col, game_board.cells))
			{
				printf("0");
			}
			else
			{
				printf(" ");
			}
		}
		printf("\n");
	}
#endif
}

bool GameBoardSectionIsDead(game_board_section* section, bool* cells)
{
	for(int i = section->start_col; i <= section->end_col; i++)
	{
		for(int j = section->start_row; j <= section->end_row; j++)
		{
			if(*GetCell(j, i, cells))
				return false;
		}
	}
	return true;
}
bool CellHasSurvived(int num_neighbors, bool current_state)
{
	switch( num_neighbors )
	{
		case 2:
			return current_state;
		case 3:
			return true;
		default:
			return false;
	}
}

bool StepGameBoardSection(game_board_section* section)
{
	int num_neighbors;
	for(int row = section->start_row; row <= section->end_row; row++)
	{
		for(int col = section->start_col; col <= section->end_col; col++)
		{
			num_neighbors = GetNumLivingNeighbors(col, row);
			*GetCell(row, col, game_board.next_cells) = CellHasSurvived(num_neighbors, *GetCell(row, col, game_board.cells));
		}
	}
	return GameBoardSectionIsDead(section, game_board.next_cells);
}

void* StepSectionThreadEntry(void* context)
{
	thread_context* ctx = (thread_context*)context;
	while(true)
	{
		while(sem_wait(&ctx->board_semaphore));
		ctx->section_dead = StepGameBoardSection(ctx->section);
		sem_post(&ctx->result_semaphore);
	}
	return NULL;
}

int main(int argv, char** argc)
{
	int size = 50;
	float percent_alive = 0.19;	
	unsigned int seed = 0;

	AllocateGameBoard(size);
	LivingEdgesInitGameBoard(size);

#ifdef RUN_MULTITHREADED
	game_board_section sections[NUM_THREADS];

	int rows_per_section = size / NUM_THREADS;
	for(int i = 0; i < NUM_THREADS; i++)
	{
		sections[i].start_col = 0;
		sections[i].end_col = size-1;
		sections[i].start_row = i * rows_per_section;
		sections[i].end_row = ((i+1) * rows_per_section)-1;
	}
	sections[NUM_THREADS-1].end_row = size-1;

	//Setup threads
	pthread_t threads[NUM_THREADS];
	thread_context thread_contexts[NUM_THREADS];
	for(int i = 0; i < NUM_THREADS; i++)
	{
		thread_contexts[i].section = &sections[i];
		sem_init(&thread_contexts[i].board_semaphore, 0, 0);
		sem_init(&thread_contexts[i].result_semaphore, 0, 0);
	}

	for(int i = 0; i < NUM_THREADS; i++)
	{
		pthread_create(&threads[i], NULL, StepSectionThreadEntry, (void*)&thread_contexts[i]);	
	}
#else
	game_board_section whole_board;
	whole_board.start_col = 0; 
	whole_board.end_col = size-1;
	whole_board.start_row = 0; 
	whole_board.end_row = size-1;
#endif

	PrintGameBoard();

	int iterations;
	int return_value;
	int* return_value_ptr = &return_value;
	start_clock();
	for(iterations = 0; iterations < 50; iterations++)
	{
#ifdef RUN_MULTITHREADED
		//post board is ready for computation
		for(int i = 0; i < NUM_THREADS; i++)
		{
			sem_post(&thread_contexts[i].board_semaphore);
		}

		//wait on results of computation
		int num_dead_sections = 0;
		for(int i = 0; i < NUM_THREADS; i++)
		{
			sem_wait(&thread_contexts[i].result_semaphore);
			num_dead_sections += thread_contexts[i].section_dead != 0;
		}
		if( num_dead_sections == NUM_THREADS )
		{
			printf("Life has ended at %d iterations\n", iterations);
			return;
		}
#else
		StepGameBoardSection(&whole_board);
#endif
		

		GameBoardFlip();
		printf("\n");
		PrintGameBoard();
#ifndef RUN_MULTITHREADED
		if(GameBoardSectionIsDead(&whole_board, game_board.cells))
			break;
#else

#endif
	}
	stop_clock();
	printf("\n\n");
	print_time_seconds(get_clock_result_seconds());
	printf("\n\n");

	return 0;
}
