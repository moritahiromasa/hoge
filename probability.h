#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TRANSITION 2    //遷移数
#define STATE 2 //状態数 (自己ループか遷移するか)
#define NUMBER 12	//ボールの個数
#define STRING 10	//文字列数

#define TSUBO_NUM 4	//ツボの数



//ボールの出力確率
void ransu_output(int ball_count[TSUBO_NUM], double output_probability[TSUBO_NUM][NUMBER])
{
		double store;
		for(int i = 0; i < TSUBO_NUM; i++)
		{
			//つぼの中にボールが一つしか入っていないとき確率を1.0にする
			if(ball_count[i] == 1)
				output_probability[i][0] = 1.0;
			
			else
			{
					
				double sum = 1.0;
				for(int j = 0; j < ball_count[i]; j++)
				{
					if(j == ball_count[i] - 1)
					{
						output_probability[i][j] = sum;
						break;
					}

					if(j == 0)
						output_probability[i][j] = pow( (double)rand() / RAND_MAX, 2.0 );

					else if(j > 0)
					{

						do{
							output_probability[i][j] = pow( (double)rand() / RAND_MAX, 2.0 );
						}while(output_probability[i][j] > sum);
					}
					
					sum -= output_probability[i][j];
				}
			}
		}
}

//自己ループか遷移するかについての確率
void ransu_state(double state_probability[TSUBO_NUM][STATE])
{
		for(int i = 0; i < TSUBO_NUM; i++)
		{
				state_probability[i][0] = pow( (double)rand() / RAND_MAX, 2.0 );
				state_probability[i][1] = 1.0 - state_probability[i][0];
		}
}

void test_output_print(int ball_count[TSUBO_NUM], double output_probability[TSUBO_NUM][NUMBER])
{
		//ボールの出力確率を表示
		for(int i = 0; i < TSUBO_NUM; i++)
		{
			printf("つぼ%d\t(", i);
			for(int j = 0; j < ball_count[i]; j++)
			{
				printf("%f", output_probability[i][j]);
				if(j != ball_count[i] - 1)
						putchar(',');
				else if(j == ball_count[i] - 1)	
						putchar(')');
			}

			putchar('\n');
		}

}

void test_state_print(int ball_count[TSUBO_NUM], double state_probability[TSUBO_NUM][STATE])
{
		//遷移するか自己ループするかについての確率
		for(int i = 0; i < TSUBO_NUM; i++)
		{
			printf("状態%d\t(", i);
			for(int j = 0; j < STATE; j++)
			{
				printf("%f", state_probability[i][j]);
				if(j == 0)
						putchar(',');
				else if(j == 1)	
						putchar(')');
			}

			putchar('\n');
		}
}

void probability_print(double state_probability[TSUBO_NUM][STATE], double output_probability[TSUBO_NUM][NUMBER], int ball_count[TSUBO_NUM])
{
		srand(time(NULL));

		ransu_output(ball_count, output_probability);
		ransu_state(state_probability);
		test_output_print(ball_count, output_probability);
		test_state_print(ball_count, state_probability);

}

