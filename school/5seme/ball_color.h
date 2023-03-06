#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TRANSITION 2	//遷移数
#define STATE 2	//状態数
#define NUMBER 12	//ボールの数
#define STRING 10	//文字列

// ボールの色を数字で識別
#define GREEN 0
#define RED 1
#define BLUE 2
#define WHITE 3

#define SPAN_1	3
#define SPAN_2	7
#define SPAN_3	11

#define TSUBO_NUM 4	//ツボの数


void Ball_color(int ball[NUMBER])
{
		for(int i = 0; i < NUMBER; i++)
		{
				if(i <= SPAN_1)
						ball[i] = GREEN;
				else if(i > SPAN_1 && i <= SPAN_2)
						ball[i] = RED;
				else if(i > SPAN_2 && i <= SPAN_3)
						ball[i] = BLUE;
				else
						ball[i] = WHITE;
		}
}

void ball_count_init(int ball_count[TSUBO_NUM])
{
		for(int i = 0; i < TSUBO_NUM; i++)
			ball_count[i] = 0;
}


// とりあえずツボの中に３つボールを入れておく
void three_put_in(int tsubo[TSUBO_NUM][NUMBER], int ball[NUMBER], int ball_count[TSUBO_NUM])
{
		int count = 0;
		for(int i = 0; i < TSUBO_NUM; i++)
		{
			if(i == 0)	
				tsubo[i][0] = rand() % NUMBER;
			else if(i > 0)
			{
				do{
					count = 0;	
					tsubo[i][0] = rand() % NUMBER;
					for(int j = 0; j < i; j++)
						if(tsubo[j][0] == tsubo[i][0])
								count++;
				}while(count != 0);
			}

			ball_count[i] += 1;
		}
}

// 残りのボールをツボの中に入れる
void the_others_ball(int tsubo[TSUBO_NUM][NUMBER], int ball[NUMBER], int ball_count[TSUBO_NUM])
{
		int n;
		
		// 残りのボールをツボの中に入れる
		for(int i = 0; i < NUMBER; i++)
		{
			int same = 0;
			
			//ボールが重複しないように印をつけておく
			for(int j = 0; j < TSUBO_NUM; j++)
				if(tsubo[j][0] == i)
					same++;			
			
			// 0から2までの乱数を生成させ、どこのツボに入れるか決める
			if(same == 0)
			{
				n = rand() % TSUBO_NUM;
				tsubo[n][ball_count[n]] = i;
				ball_count[n] += 1;
			}
			
		}

}

void test_print(int tsubo[TSUBO_NUM][NUMBER], int ball[NUMBER], int ball_count[TSUBO_NUM])
{
		for(int i = 0; i < TSUBO_NUM; i++)
		{
			printf("つぼ%dの個数:%3d(", i, ball_count[i]);
			
			for(int j = 0; j < ball_count[i]; j++)
			{
				printf("%2d", tsubo[i][j]);
				if(j != ( ball_count[i] - 1) )
					putchar(',');
				else if(j == ( ball_count[i] - 1) )
					putchar(')');
			}
			putchar('\n');
		}
}


