#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STATE 2	//状態数(自己ループか遷移するか)
#define NUMBER 12	//ボールの個数


#define TSUBO_NUM 4	//ツボの数

#define OUTPUT_LOOP_RANSU 6	//ループの経路6通り分
#define	STATE_LOOP_RANSU 5	//５通り分でどのボールが出るかを決める

#define ARRAY_NUM 20	// 配列数20個分

int a[OUTPUT_LOOP_RANSU][ARRAY_NUM] = { 
												{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 1, 2, 2},
												{0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2},
												{0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 0, 2, 2, 2, 1, 2, 2},
												{0, 1, 0, 0, 1, 1, 1, 1, 2, 0, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2},
												{0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 1, 2, 2},
												{0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2}
											};

int Ransu(void)
{
		return rand() % OUTPUT_LOOP_RANSU;
}

// 自己ループは0、遷移は1とする
void loop_keisan(double wa, int i, int ball_count[NUMBER], double state_probability[TSUBO_NUM][STATE], double output_probability[TSUBO_NUM][NUMBER])
{
		i  = Ransu();	//ループ経路を決める変数

		wa = state_probability[a[i][0]][a[i][1]] * output_probability[a[i][2]][ rand() % ball_count[a[i][3]] ] + state_probability[a[i][4]][a[i][5]] * output_probability[a[i][6]][ rand() % ball_count[a[i][7]] ] + state_probability[a[i][8]][a[i][9]] * output_probability[a[i][10]][ rand() % ball_count[a[i][11]] ] + state_probability[a[i][12]][a[i][13]] * output_probability[a[i][14]][ rand() % ball_count[a[i][15]] ] + state_probability[a[i][16]][a[i][17]] * output_probability[a[i][18]][ rand() % ball_count[a[i][19]] ];
		printf("%d番の経路\n", i);
		printf("wa = %lf\n", wa);
}


void loop(double wa, int state_loop_ransu, int ball_count[NUMBER], double state_probability[TSUBO_NUM][STATE], double output_probability[TSUBO_NUM][NUMBER])
{

		loop_keisan(wa, state_loop_ransu, ball_count, state_probability, output_probability);
}




