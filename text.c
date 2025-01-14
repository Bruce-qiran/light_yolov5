#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义学生结构体
struct Student
{
    char id[15];   // 学号，14位整数
    int scores[6]; // 各科成绩，依次为语文、数学、英语、物理、化学、生物
    int ori;       // 原数据中的位置
};
int main()
{
    int n, k;
    // 读取学生数量
    scanf("%d", &n);
    // 创建学生数组
    struct Student students[n];

    // 读取学生数据
    for (int i = 0; i < n; i++)
    {
        scanf("%s", students[i].id);
        for (int j = 0; j < 6; j++)
        {
            scanf("%d", &students[i].scores[j]);
        }
        students[i].originalIndex = i + 1; // 保存原始位置
    }

    // 读取排序关键字（科目索引）
    scanf("%d", &k);
    k--; // 将科目转换为索引（1~6 转换为 0~5）

    // 排序
    qsort_r(students, n, sizeof(struct Student), compare, &k);

    // 输出排序结果，输出每个学生原数据中的位置
    for (int i = 0; i < n; i++)
    {
        printf("%d\n", students[i].originalIndex);
    }

    return 0;
}
