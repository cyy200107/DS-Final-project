#!/usr/bin/env Rscript

##################
# soccer_modeling.R
# 足球比赛建模分析
##################

# 设置日志文件
log_file <- file.path("logs", format(Sys.time(), "modeling_log_%Y%m%d_%H%M%S.txt"))
dir.create("logs", showWarnings = FALSE)

#' 日志函数
log_message <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message <- sprintf("[%s] %s: %s", timestamp, level, msg)
  cat(message, "\n", file = log_file, append = TRUE)
  if(level %in% c("ERROR", "WARNING")) {
    cat(message, "\n")
  }
}

#' 加载必要的包
load_packages <- function() {
  # 设置CRAN镜像
  options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
  
  packages <- c(
    "tidyverse", "caret", "glmnet", "brms", "lme4", "MASS", 
    "geepack", "performance", "MatchIt", "nnet", "ggplot2", 
    "gridExtra", "fmsb", "rmarkdown", "dplyr", "tidyr"
  )
  
  for(pkg in packages) {
    tryCatch({
      if(!requireNamespace(pkg, quietly = TRUE)) {
        log_message(sprintf("Installing package: %s", pkg))
        install.packages(pkg)
      }
      library(pkg, character.only = TRUE)
      log_message(sprintf("Successfully loaded package: %s", pkg))
    }, error = function(e) {
      log_message(sprintf("Failed to load package %s: %s", pkg, e$message), "ERROR")
      stop(sprintf("Failed to load package %s", pkg))
    })
  }
}

#' 加载和验证数据
load_and_validate <- function() {
  # 检查处理后的数据文件是否存在
  data_files <- list(
    all = "processed_data/all_leagues_processed.rds",
    bundesliga = "processed_data/bundesliga_processed.rds",
    laliga = "processed_data/laliga_processed.rds", 
    ligue1 = "processed_data/ligue1_processed.rds",
    premier = "processed_data/premier_processed.rds",
    seriea = "processed_data/seriea_processed.rds"
  )
  
  data_list <- list()
  
  for(name in names(data_files)) {
    file_path <- data_files[[name]]
    tryCatch({
      if(file.exists(file_path)) {
        data_list[[name]] <- readRDS(file_path)
        log_message(sprintf("Successfully loaded data for %s", name))
      } else {
        log_message(sprintf("File not found: %s", file_path), "WARNING")
      }
    }, error = function(e) {
      log_message(sprintf("Error loading %s: %s", name, e$message), "ERROR")
    })
  }
  
  if(length(data_list) == 0) {
    stop("No data files could be loaded")
  }
  
  return(data_list)
}

#' No Polling Models
fit_no_polling <- function(data, leagues) {
  models <- list()
  
  # 检查必需的变量是否存在
  required_vars <- c("League", "TotalGoals", "HomeResult", "GoalDiff",
                     "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
                     "HomeAttackEff", "AwayAttackEff", "HomeDominance")
  
  missing_vars <- setdiff(required_vars, names(data))
  if(length(missing_vars) > 0) {
    stop(sprintf("Missing required variables: %s", 
                 paste(missing_vars, collapse = ", ")))
  }
  
  for(league in leagues) {
    log_message(sprintf("Fitting models for league: %s", league))
    
    league_data <- data %>% 
      filter(League == league)
    
    if(nrow(league_data) == 0) {
      log_message(sprintf("No data available for league: %s", league), "WARNING")
      next
    }
    
    tryCatch({
      # 1. 总进球数预测（使用泊松回归）
      goals_formula <- TotalGoals ~ HS + AS + HST + AST + HF + AF + HC + AC +
        HomeAttackEff + AwayAttackEff + HomeDominance
      
      goals_model <- glm(goals_formula,
                         family = poisson(link = "log"),
                         data = league_data,
                         control = list(maxit = 100))
      
      # 2. 比赛结果预测（使用多项式逻辑回归）
      result_formula <- HomeResult ~ HS + AS + HST + AST + HF + AF + HC + AC +
        HomeAttackEff + AwayAttackEff + HomeDominance
      
      result_model <- multinom(result_formula,
                               data = league_data,
                               trace = FALSE,
                               maxit = 1000)
      
      # 3. 进球差预测（使用线性回归）
      diff_formula <- GoalDiff ~ HS + AS + HST + AST + HF + AF + HC + AC +
        HomeAttackEff + AwayAttackEff + HomeDominance
      
      diff_model <- lm(diff_formula, data = league_data)
      
      models[[league]] <- list(
        goals = goals_model,
        result = result_model,
        diff = diff_model
      )
      
      log_message(sprintf("Successfully fitted models for league: %s", league))
      
    }, error = function(e) {
      log_message(sprintf("Error in model fitting for league %s: %s", 
                          league, e$message), "ERROR")
      return(NULL)
    })
  }
  
  if(length(models) == 0) {
    log_message("No models could be fitted", "WARNING")
  }
  
  return(models)
}

#' Partial Polling Models
fit_partial_polling <- function(data) {
  log_message("Starting partial polling models fitting")
  
  # 数据预处理
  data <- data %>%
    group_by(League) %>%
    mutate(
      AttackingStyle = mean(TotalGoals, na.rm = TRUE) > median(TotalGoals, na.rm = TRUE),
      LeagueStrength = mean(TotalShots, na.rm = TRUE) > median(TotalShots, na.rm = TRUE),
      LeagueType = factor(paste0(
        ifelse(AttackingStyle, "High", "Low"), "_",
        ifelse(LeagueStrength, "Strong", "Weak")
      )),
      match_id = row_number(),
      time_index = match_id
    ) %>%
    ungroup()
  
  tryCatch({
    # 1. 多层泊松回归（进球数）
    log_message("Fitting multilevel Poisson regression...")
    mlm_goals <- glmer(
      TotalGoals ~ HS + AS + (1|League),
      family = poisson(link = "log"),
      data = data,
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
    )
    
    # 2. 多层多项式逻辑回归（比赛结果）
    log_message("Fitting multilevel multinomial regression...")
    mlm_result <- multinom(
      HomeResult ~ HS + AS + League,
      data = data,
      trace = FALSE,
      MaxNWts = 5000
    )
    
    # 3. GEE模型（考虑时间相关性）
    log_message("Fitting GEE model...")
    data$id <- as.numeric(factor(data$League))
    data$wave <- as.numeric(factor(data$match_id))
    
    gee_model <- geeglm(
      TotalGoals ~ HS + AS,
      id = id,
      wave = wave,
      data = data,
      family = poisson(link = "log"),
      corstr = "exchangeable"
    )
    
    log_message("Successfully fitted all partial polling models")
    
    return(list(
      mlm_goals = mlm_goals,
      mlm_result = mlm_result,
      gee = gee_model
    ))
    
  }, error = function(e) {
    log_message(sprintf("Error in partial polling models: %s", e$message), "ERROR")
    return(NULL)
  })
}

#' Complete Polling Models
fit_complete_polling <- function(data) {
  log_message("Starting complete polling models fitting")
  
  tryCatch({
    # 1. 泊松回归（总进球数）
    log_message("Fitting Poisson regression...")
    complete_goals <- glm(
      TotalGoals ~ HS + AS + HST + AST + League,
      family = poisson(),
      data = data,
      control = list(maxit = 1000)
    )
    
    # 2. 多项式逻辑回归（比赛结果）
    log_message("Fitting multinomial regression...")
    complete_result <- multinom(
      HomeResult ~ HS + AS + HST + AST + League,
      data = data,
      trace = FALSE,
      MaxNWts = 5000
    )
    
    # 3. 贝叶斯多层模型
    log_message("Fitting Bayesian multilevel model...")
    bayes_model <- brm(
      formula = TotalGoals ~ HS + AS + HST + AST + (1|League),
      data = data,
      family = poisson(),
      prior = c(
        prior(normal(0, 5), class = "b"),
        prior(normal(0, 2), class = "Intercept"),
        prior(student_t(3, 0, 2), class = "sd")
      ),
      chains = 4,
      iter = 2000,
      warmup = 1000,
      cores = 4,
      control = list(adapt_delta = 0.99, max_treedepth = 15)
    )
    
    log_message("Successfully fitted all complete polling models")
    
    return(list(
      goals = complete_goals,
      result = complete_result,
      bayes = bayes_model
    ))
    
  }, error = function(e) {
    log_message(sprintf("Error in complete polling models: %s", e$message), "ERROR")
    return(NULL)
  })
}

#' 计算Kappa系数
calculate_kappa <- function(actual, predicted) {
  # 创建混淆矩阵
  conf_matrix <- table(actual, predicted)
  
  # 计算观察到的一致性
  observed_agreement <- sum(diag(conf_matrix)) / sum(conf_matrix)
  
  # 计算期望的一致性
  row_sums <- rowSums(conf_matrix)
  col_sums <- colSums(conf_matrix)
  expected_agreement <- sum(row_sums * col_sums) / (sum(conf_matrix)^2)
  
  # 计算 Kappa
  kappa <- (observed_agreement - expected_agreement) / (1 - expected_agreement)
  
  return(kappa)
}

#' 模型评估
evaluate_models <- function(no_polling, partial_polling, complete_polling, data) {
  results <- list()
  
  # 1. 评估No Polling模型
  log_message("Evaluating No Polling models...")
  no_polling_metrics <- list()
  
  for(league in names(no_polling)) {
    if(!is.null(no_polling[[league]])) {
      metrics <- list()
      tryCatch({
        # 进球数预测评估
        pred_goals <- predict(no_polling[[league]]$goals, newdata = data, type = "response")
        metrics$goals <- c(
          RMSE = sqrt(mean((data$TotalGoals - pred_goals)^2, na.rm = TRUE)),
          MAE = mean(abs(data$TotalGoals - pred_goals), na.rm = TRUE)
        )
        
        # 比赛结果预测评估
        pred_result <- predict(no_polling[[league]]$result, newdata = data)
        metrics$result <- c(
          Accuracy = mean(pred_result == data$HomeResult, na.rm = TRUE),
          Kappa = calculate_kappa(data$HomeResult, pred_result)
        )
        
        no_polling_metrics[[league]] <- metrics
        
      }, error = function(e) {
        log_message(sprintf("Error in evaluating no polling model for %s: %s", league, e$message), "WARNING")
      })
    }
  }
  
  results$no_polling <- no_polling_metrics
  
  # 2. 评估Partial Polling模型
  if(!is.null(partial_polling)) {
    partial_metrics <- list()
    
    tryCatch({
      if(!is.null(partial_polling$mlm_goals)) {
        pred_goals <- predict(partial_polling$mlm_goals, newdata = data, 
                              re.form = NULL)
        partial_metrics$goals <- c(
          RMSE = sqrt(mean((data$TotalGoals - pred_goals)^2, na.rm = TRUE)),
          MAE = mean(abs(data$TotalGoals - pred_goals), na.rm = TRUE)
        )
      }
      
      if(!is.null(partial_polling$mlm_result)) {
        pred_result <- predict(partial_polling$mlm_result, newdata = data)
        partial_metrics$result <- c(
          Accuracy = mean(pred_result == data$HomeResult, na.rm = TRUE),
          Kappa = calculate_kappa(data$HomeResult, pred_result)
        )
      }
      
      results$partial_polling <- partial_metrics
      
    }, error = function(e) {
      log_message(sprintf("Error in partial polling evaluation: %s", e$message), "WARNING")
    })
  }
  
  # 3. 评估Complete Polling模型
  if(!is.null(complete_polling)) {
    complete_metrics <- list()
    
    tryCatch({
      # 评估泊松回归
      if(!is.null(complete_polling$goals)) {
        pred_goals <- predict(complete_polling$goals, newdata = data, 
                              type = "response")
        complete_metrics$goals <- c(
          RMSE = sqrt(mean((data$TotalGoals - pred_goals)^2, na.rm = TRUE)),
          MAE = mean(abs(data$TotalGoals - pred_goals), na.rm = TRUE)
        )
      }
      
      # 评估多项式逻辑回归
      if(!is.null(complete_polling$result)) {
        pred_result <- predict(complete_polling$result, newdata = data)
        complete_metrics$result <- c(
          Accuracy = mean(pred_result == data$HomeResult, na.rm = TRUE),
          Kappa = calculate_kappa(data$HomeResult, pred_result)
        )
      }
      
      # 评估贝叶斯模型
      if(!is.null(complete_polling$bayes)) {
        pred_bayes <- fitted(complete_polling$bayes)  # 使用fitted()而不是predict()
        complete_metrics$bayes <- c(
          RMSE = sqrt(mean((data$TotalGoals - pred_bayes[,1])^2, na.rm = TRUE)),
          MAE = mean(abs(data$TotalGoals - pred_bayes[,1]), na.rm = TRUE)
        )
      }
      
      results$complete_polling <- complete_metrics
      
    }, error = function(e) {
      log_message(sprintf("Error in complete polling evaluation: %s", e$message), "WARNING")
    })
  }
  
  return(results)
}

#' 创建结果摘要
create_summary <- function(eval_results) {
  # 创建空数据框
  summary_df <- data.frame(
    model_type = character(),
    goals_rmse = numeric(),
    goals_mae = numeric(),
    result_accuracy = numeric(),
    result_kappa = numeric(),
    bayes_rmse = numeric(),
    bayes_mae = numeric(),
    stringsAsFactors = FALSE
  )
  
  # 处理No Polling结果
  if(!is.null(eval_results$no_polling)) {
    goals_metrics <- sapply(eval_results$no_polling, function(x) x$goals)
    result_metrics <- sapply(eval_results$no_polling, function(x) x$result)
    
    no_polling_summary <- data.frame(
      model_type = "no_polling",
      goals_rmse = mean(unlist(goals_metrics["RMSE",]), na.rm = TRUE),
      goals_mae = mean(unlist(goals_metrics["MAE",]), na.rm = TRUE),
      result_accuracy = mean(unlist(result_metrics["Accuracy",]), na.rm = TRUE),
      result_kappa = mean(unlist(result_metrics["Kappa",]), na.rm = TRUE),
      bayes_rmse = NA,
      bayes_mae = NA
    )
    summary_df <- rbind(summary_df, no_polling_summary)
  }
  
  # 处理Partial Polling结果
  if(!is.null(eval_results$partial_polling)) {
    partial_polling_summary <- data.frame(
      model_type = "partial_polling",
      goals_rmse = eval_results$partial_polling$goals["RMSE"],
      goals_mae = eval_results$partial_polling$goals["MAE"],
      result_accuracy = eval_results$partial_polling$result["Accuracy"],
      result_kappa = eval_results$partial_polling$result["Kappa"],
      bayes_rmse = NA,
      bayes_mae = NA
    )
    summary_df <- rbind(summary_df, partial_polling_summary)
  }
  
  # 处理Complete Polling结果
  if(!is.null(eval_results$complete_polling)) {
    complete_polling_summary <- data.frame(
      model_type = "complete_polling",
      goals_rmse = eval_results$complete_polling$goals["RMSE"],
      goals_mae = eval_results$complete_polling$goals["MAE"],
      result_accuracy = eval_results$complete_polling$result["Accuracy"],
      result_kappa = eval_results$complete_polling$result["Kappa"],
      bayes_rmse = eval_results$complete_polling$bayes["RMSE"],
      bayes_mae = eval_results$complete_polling$bayes["MAE"]
    )
    summary_df <- rbind(summary_df, complete_polling_summary)
  }
  
  rownames(summary_df) <- summary_df$model_type
  summary_df$model_type <- NULL
  
  return(summary_df)
}

#' 创建联赛对比分析
create_league_comparison <- function(data) {
  # 创建保存图片的目录
  if(!dir.exists("analysis_results/figures")) {
    dir.create("analysis_results/figures", recursive = TRUE)
  }
  
  # 1. 进球分布分析
  goals_plot <- ggplot(data, aes(x = League, y = TotalGoals, fill = League)) +
    geom_boxplot() +
    theme_minimal() +
    labs(title = "进球分布对比",
         x = "联赛",
         y = "总进球数") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("analysis_results/figures/goals_distribution.png", goals_plot, width = 10, height = 6)
  
  # 2. 主场优势分析
  home_advantage <- data %>%
    group_by(League) %>%
    summarise(
      home_win_rate = mean(HomeResult == "Win", na.rm = TRUE),
      home_goals_per_game = mean(FTHG, na.rm = TRUE),
      away_goals_per_game = mean(FTAG, na.rm = TRUE)
    )
  
  home_adv_plot <- ggplot(home_advantage, aes(x = League)) +
    geom_col(aes(y = home_win_rate, fill = "主场胜率")) +
    geom_point(aes(y = home_goals_per_game, color = "主场场均进球")) +
    geom_point(aes(y = away_goals_per_game, color = "客场场均进球")) +
    theme_minimal() +
    labs(title = "主场优势分析",
         x = "联赛",
         y = "比率") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("analysis_results/figures/home_advantage.png", home_adv_plot, width = 10, height = 6)
  
  # 3. 攻防能力雷达图
  league_stats <- data %>%
    group_by(League) %>%
    summarise(
      avg_goals = mean(TotalGoals, na.rm = TRUE),
      shots_accuracy = mean(HST/HS, na.rm = TRUE),
      shot_conversion = mean(FTHG/HST, na.rm = TRUE),
      possession = mean(HomeDominance, na.rm = TRUE),
      corner_kicks = mean(HC, na.rm = TRUE)
    )
  
  radar_data <- as.data.frame(t(league_stats[,-1]))
  colnames(radar_data) <- league_stats$League
  radar_data <- rbind(apply(radar_data, 2, max),
                      apply(radar_data, 2, min),
                      radar_data)
  
  png("analysis_results/figures/radar_chart.png", width = 800, height = 800)
  radarchart(radar_data,
             pcol = rainbow(ncol(radar_data)),
             title = "联赛攻防能力对比")
  legend("topright",
         legend = colnames(radar_data),
         col = rainbow(ncol(radar_data)),
         lty = 1)
  dev.off()
  
  # 4. 球队实力分布
  team_strength <- data %>%
    group_by(League, HomeTeam) %>%
    summarise(
      avg_goals_scored = mean(FTHG, na.rm = TRUE),
      avg_goals_conceded = mean(FTAG, na.rm = TRUE),
      points = sum(case_when(
        HomeResult == "Win" ~ 3,
        HomeResult == "Draw" ~ 1,
        TRUE ~ 0
      )),
      .groups = 'drop'
    )
  
  strength_plot <- ggplot(team_strength, 
                         aes(x = avg_goals_scored, 
                             y = avg_goals_conceded, 
                             color = League)) +
    geom_point() +
    theme_minimal() +
    labs(title = "球队实力分布",
         x = "场均进球",
         y = "场均失球")
  
  ggsave("analysis_results/figures/team_strength.png", strength_plot, width = 10, height = 6)
  
  return(list(
    goals_stats = summary(data$TotalGoals),
    home_advantage = home_advantage,
    league_stats = league_stats,
    team_strength = team_strength
  ))
}

#' 创建时间趋势分析
create_temporal_analysis <- function(data) {
  # 确保日期格式正确
  data$Date <- as.Date(data$Date)
  
  # 1. 按月进球趋势
  monthly_goals <- data %>%
    mutate(Month = format(Date, "%Y-%m")) %>%
    group_by(League, Month) %>%
    summarise(
      avg_goals = mean(TotalGoals, na.rm = TRUE),
      .groups = 'drop'
    )
  
  goals_trend_plot <- ggplot(monthly_goals, 
                            aes(x = Month, y = avg_goals, 
                                color = League, group = League)) +
    geom_line() +
    theme_minimal() +
    labs(title = "月度进球趋势",
         x = "月份",
         y = "平均进球数") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave("analysis_results/figures/monthly_goals_trend.png", 
         goals_trend_plot, width = 12, height = 6)
  
  # 2. 赛季疲劳分析
  fatigue_analysis <- data %>%
    mutate(Month = format(Date, "%m")) %>%
    group_by(League, Month) %>%
    summarise(
      fouls_per_game = mean(HF + AF, na.rm = TRUE),
      cards_per_game = mean(HY + AY + HR + AR, na.rm = TRUE),
      goals_per_game = mean(TotalGoals, na.rm = TRUE),
      shots_per_game = mean(HS + AS, na.rm = TRUE),
      .groups = 'drop'
    )
  
  # 绘制综合疲劳指标图
  fatigue_plot <- ggplot(fatigue_analysis) +
    geom_line(aes(x = Month, y = scale(fouls_per_game), color = League)) +
    geom_line(aes(x = Month, y = scale(cards_per_game), linetype = "Cards")) +
    facet_wrap(~League) +
    theme_minimal() +
    labs(title = "赛季疲劳指标趋势",
         x = "月份",
         y = "标准化指标值")
  
  ggsave("analysis_results/figures/fatigue_trends.png", 
         fatigue_plot, width = 15, height = 10)
  
  return(list(
    monthly_goals = monthly_goals,
    fatigue_analysis = fatigue_analysis
  ))
}

#' 生成详细的Markdown报告
generate_markdown_report <- function(analysis_results, data) {
  # 计算一些基础统计量
  league_summaries <- data %>%
    group_by(League) %>%
    summarise(
      total_matches = n(),
      total_goals = sum(TotalGoals),
      avg_goals = mean(TotalGoals, na.rm = TRUE),
      home_win_rate = mean(HomeResult == "Win", na.rm = TRUE),
      draw_rate = mean(HomeResult == "Draw", na.rm = TRUE),
      away_win_rate = mean(HomeResult == "Loss", na.rm = TRUE),
      avg_shots = mean(HS + AS, na.rm = TRUE),
      avg_shots_on_target = mean(HST + AST, na.rm = TRUE),
      avg_corners = mean(HC + AC, na.rm = TRUE),
      avg_fouls = mean(HF + AF, na.rm = TRUE),
      avg_cards = mean(HY + AY + HR + AR, na.rm = TRUE)
    )

  # 比赛特征分析
  match_patterns <- data %>%
    mutate(
      high_scoring = TotalGoals >= 4,
      low_scoring = TotalGoals <= 1,
      clean_sheet = FTHG == 0 | FTAG == 0,
      comeback_win = (HTHG < HTAG & FTR == "H") | (HTHG > HTAG & FTR == "A")
    ) %>%
    group_by(League) %>%
    summarise(
      high_scoring_rate = mean(high_scoring, na.rm = TRUE),
      low_scoring_rate = mean(low_scoring, na.rm = TRUE),
      clean_sheet_rate = mean(clean_sheet, na.rm = TRUE),
      comeback_rate = mean(comeback_win, na.rm = TRUE)
    )

  report <- c(
    "# 五大联赛足球数据深度分析报告\n\n",
    
    "## 目录\n",
    "1. [联赛整体概况](#联赛整体概况)\n",
    "2. [进球分析](#进球分析)\n",
    "3. [主场优势分析](#主场优势分析)\n",
    "4. [比赛模式分析](#比赛模式分析)\n",
    "5. [技术指标分析](#技术指标分析)\n",
    "6. [联赛特征对比](#联赛特征对比)\n",
    "7. [时间趋势分析](#时间趋势分析)\n",
    "8. [特殊比赛模式](#特殊比赛模式)\n",
    "9. [总结与洞察](#总结与洞察)\n\n",

    "## 联赛整体概况\n\n",
    
    paste(sapply(1:nrow(league_summaries), function(i) {
      league <- league_summaries[i,]
      sprintf(
        "### %s\n\n- 总场次：%d\n- 总进球：%d\n- 场均进球：%.2f\n- 主场胜率：%.1f%%\n- 平局率：%.1f%%\n- 客场胜率：%.1f%%\n\n",
        league$League,
        league$total_matches,
        league$total_goals,
        league$avg_goals,
        league$home_win_rate * 100,
        league$draw_rate * 100,
        league$away_win_rate * 100
      )
    }), collapse = "\n"),

    "## 进球分析\n\n",
    
    "### 进球分布\n\n",
    "![进球分布](figures/goals_distribution.png)\n\n",
    
    paste(sapply(1:nrow(league_summaries), function(i) {
      league <- league_summaries[i,]
      sprintf(
        "- %s的进球特征：\n  - 场均%.2f个进球\n  - %.1f%%的比赛为高进球(4+)场次\n  - %.1f%%的比赛为低进球(0-1)场次\n",
        league$League,
        league$avg_goals,
        match_patterns[i,]$high_scoring_rate * 100,
        match_patterns[i,]$low_scoring_rate * 100
      )
    }), collapse = "\n"),
    
    "\n## 主场优势分析\n\n",
    "![主场优势](figures/home_advantage.png)\n\n",
    
    paste(sapply(1:nrow(league_summaries), function(i) {
      league <- league_summaries[i,]
      sprintf(
        "- %s的主场优势：\n  - 主场胜率：%.1f%%\n  - 主场进球效率：%.2f个/场\n  - 主场控球率：%.1f%%\n",
        league$League,
        league$home_win_rate * 100,
        league$avg_goals,
        mean(data$HomeDominance[data$League == league$League], na.rm = TRUE) * 100
      )
    }), collapse = "\n"),
    
    "\n## 技术指标分析\n\n",
    "![攻防能力](figures/radar_chart.png)\n\n",
    
    paste(sapply(1:nrow(league_summaries), function(i) {
      league <- league_summaries[i,]
      sprintf(
        "- %s的技术特征：\n  - 场均射门：%.1f次\n  - 射正率：%.1f%%\n  - 角球数：%.1f个\n  - 犯规数：%.1f次\n",
        league$League,
        league$avg_shots,
        (league$avg_shots_on_target / league$avg_shots) * 100,
        league$avg_corners,
        league$avg_fouls
      )
    }), collapse = "\n"),
    
    "\n## 时间趋势分析\n\n",
    "### 赛季进程中的变化\n",
    "![月度进球趋势](figures/monthly_goals_trend.png)\n\n",
    "![疲劳指标](figures/fatigue_trends.png)\n\n",
    
    "### 赛季疲劳分析\n",
    sprintf(
      "- 犯规数最高的赛季阶段：%s\n",
      names(which.max(tapply(data$HF + data$AF, 
                            format(as.Date(data$Date), "%B"), 
                            mean, na.rm = TRUE)))
    ),
    sprintf(
      "- 进球数最高的赛季阶段：%s\n",
      names(which.max(tapply(data$TotalGoals, 
                            format(as.Date(data$Date), "%B"), 
                            mean, na.rm = TRUE)))
    ),
    
    "\n## 特殊比赛模式\n\n",
    paste(sapply(1:nrow(match_patterns), function(i) {
      pattern <- match_patterns[i,]
      sprintf(
        "- %s：\n  - 零封率：%.1f%%\n  - 大比分率：%.1f%%\n  - 逆转率：%.1f%%\n",
        league_summaries$League[i],
        pattern$clean_sheet_rate * 100,
        pattern$high_scoring_rate * 100,
        pattern$comeback_rate * 100
      )
    }), collapse = "\n"),
    
    "\n## 总结与洞察\n\n",
    
    "### 联赛特点总结\n",
    paste(sapply(unique(data$League), function(league) {
      league_data <- data[data$League == league,]
      sprintf(
        "- %s：\n  - 特点：%s\n  - 优势：%s\n  - 挑战：%s\n",
        league,
        ifelse(mean(league_data$TotalGoals) > mean(data$TotalGoals), "进攻型联赛", "防守型联赛"),
        ifelse(mean(league_data$HomeResult == "Win") > mean(data$HomeResult == "Win"), "主场优势明显", "主客场平衡"),
        ifelse(sd(league_data$TotalGoals) > sd(data$TotalGoals), "比赛结果不稳定", "比赛结果稳定")
      )
    }), collapse = "\n"),
    
    "\n### 关键发现\n",
    "1. 进球特征：\n",
    sprintf("   - 最高进球效率联赛：%s\n",
            league_summaries$League[which.max(league_summaries$avg_goals)]),
    sprintf("   - 最稳定联赛：%s\n",
            league_summaries$League[which.min(sapply(split(data$TotalGoals, data$League), sd, na.rm = TRUE))]),
    "2. 比赛模式：\n",
    sprintf("   - 最具观赏性联赛：%s（基于高进球率和射门数）\n",
            league_summaries$League[which.max(league_summaries$avg_shots)]),
    sprintf("   - 最激烈对抗联赛：%s（基于犯规数和黄牌数）\n",
            league_summaries$League[which.max(league_summaries$avg_fouls)]),
    "3. 特殊发现：\n",
    sprintf("   - 逆转率最高联赛：%s\n",
            match_patterns$League[which.max(match_patterns$comeback_rate)]),
    sprintf("   - 零封率最高联赛：%s\n",
            match_patterns$League[which.max(match_patterns$clean_sheet_rate)]),
    
    "\n### 建议与展望\n",
    "1. 联赛发展建议：\n",
    "   - 加强进攻战术的培养\n",
    "   - 平衡主客场优势\n",
    "   - 提高比赛观赏性\n",
    "2. 数据应用建议：\n",
    "   - 深入分析球队战术特点\n",
    "   - 优化赛程安排\n",
    "   - 改进裁判执法标准\n",
    "3. 未来研究方向：\n",
    "   - 球员表现数据分析\n",
    "   - 战术趋势研究\n",
    "   - 跨赛季比较研究\n"
  )
  
  # 写入markdown文件
  writeLines(report, "analysis_results/analysis_report.md")
  
  # 生成HTML版本（如果安装了rmarkdown包）
  if(requireNamespace("rmarkdown", quietly = TRUE)) {
    tryCatch({
      rmarkdown::render("analysis_results/analysis_report.md",
                       output_format = "html_document",
                       output_file = "analysis_report.html")
    }, error = function(e) {
      warning("HTML report generation failed: ", e$message)
    })
  }
}

#' 主函数
#' 主函数
main <- function() {
  # 设置随机数种子以确保可重复性
  set.seed(123)
  
  # 创建必要的目录
  dirs <- c("logs", "model_results", "analysis_results", "analysis_results/figures")
  for(dir in dirs) {
    if(!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE)
      log_message(sprintf("Created directory: %s", dir))
    }
  }
  
  # 加载必要的包
  load_packages()
  
  # 加载数据
  tryCatch({
    log_message("Loading and validating data...")
    data_list <- load_and_validate()
    
    # 检查数据是否成功加载
    if(is.null(data_list$all)) {
      stop("Failed to load all leagues data")
    }
    
    # 获取联赛列表
    leagues <- unique(data_list$all$League)
    
    if(length(leagues) == 0) {
      stop("No league data available")
    }
    
    # === 建模分析部分 ===
    
    # 1. No Polling Models
    log_message("Fitting No Polling models...")
    no_polling_models <- fit_no_polling(data_list$all, leagues)
    
    # 2. Partial Polling Models
    log_message("Fitting Partial Polling models...")
    partial_polling_models <- fit_partial_polling(data_list$all)
    
    # 3. Complete Polling Models
    log_message("Fitting Complete Polling models...")
    complete_polling_models <- fit_complete_polling(data_list$all)
    
    # 评估模型
    log_message("Evaluating models...")
    eval_results <- evaluate_models(
      no_polling_models,
      partial_polling_models,
      complete_polling_models,
      data_list$all
    )
    
    # 创建结果摘要
    log_message("Creating results summary...")
    summary_results <- create_summary(eval_results)
    
    # === 多维度分析部分 ===
    
    log_message("Starting comprehensive analysis...")
    
    # 计算联赛汇总统计
    league_summaries <- data_list$all %>%
      group_by(League) %>%
      summarise(
        total_matches = n(),
        total_goals = sum(TotalGoals),
        avg_goals = mean(TotalGoals, na.rm = TRUE),
        home_win_rate = mean(HomeResult == "Win", na.rm = TRUE),
        draw_rate = mean(HomeResult == "Draw", na.rm = TRUE),
        away_win_rate = mean(HomeResult == "Loss", na.rm = TRUE),
        avg_shots = mean(HS + AS, na.rm = TRUE),
        avg_shots_on_target = mean(HST + AST, na.rm = TRUE),
        avg_corners = mean(HC + AC, na.rm = TRUE),
        avg_fouls = mean(HF + AF, na.rm = TRUE),
        avg_cards = mean(HY + AY + HR + AR, na.rm = TRUE)
      )
    
    # 计算比赛模式统计
    match_patterns <- data_list$all %>%
      mutate(
        high_scoring = TotalGoals >= 4,
        low_scoring = TotalGoals <= 1,
        clean_sheet = FTHG == 0 | FTAG == 0,
        comeback_win = (HTHG < HTAG & FTR == "H") | (HTHG > HTAG & FTR == "A")
      ) %>%
      group_by(League) %>%
      summarise(
        high_scoring_rate = mean(high_scoring, na.rm = TRUE),
        low_scoring_rate = mean(low_scoring, na.rm = TRUE),
        clean_sheet_rate = mean(clean_sheet, na.rm = TRUE),
        comeback_rate = mean(comeback_win, na.rm = TRUE)
      )
    
    # 1. 联赛对比分析
    log_message("Performing league comparison analysis...")
    comparison_results <- create_league_comparison(data_list$all)
    
    # 2. 时间趋势分析
    log_message("Performing temporal analysis...")
    temporal_results <- create_temporal_analysis(data_list$all)
    
    # === 结果汇总和报告生成 ===
    
    # 1. 生成分析报告
    log_message("Generating analysis report...")
    analysis_results <- list(
      comparison = comparison_results,
      temporal = temporal_results,
      league_summaries = league_summaries,
      match_patterns = match_patterns
    )
    generate_markdown_report(analysis_results, data_list$all)
    
    # 2. 保存分析结果
    log_message("Saving analysis results...")
    saveRDS(analysis_results, "analysis_results/analysis_results.rds")
    
    # 3. 保存模型结果
    log_message("Saving modeling results...")
    saveRDS(list(
      no_polling = no_polling_models,
      partial_polling = partial_polling_models,
      complete_polling = complete_polling_models
    ), "model_results/model_objects.rds")
    
    # 4. 保存评估结果
    saveRDS(eval_results, "model_results/evaluation_results.rds")
    write.csv(summary_results, "model_results/model_summary.csv")
    
    log_message("Analysis completed successfully")
    
  }, error = function(e) {
    log_message(sprintf("Error in main execution: %s", e$message), "ERROR")
    stop("Analysis failed")
  })
}

# 运行主函数
if(!interactive()) {
  tryCatch({
    main()
  }, error = function(e) {
    log_message(sprintf("Critical error in main execution: %s", e$message), "ERROR")
    quit(status = 1)
  })
}