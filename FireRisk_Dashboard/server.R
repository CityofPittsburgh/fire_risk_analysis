# Data dashboard for Metro21 Fire Risk Analysis Project
# Created for: Pittsburgh Bureau of Fire
# Authors: Qianyi Hu, Michael Madaio, Geoffrey Arnold
# Latest update: February 20, 2018


# The server side of the dashboard

source("global.R", local = FALSE)

shinyServer(function(input, output, session) {
  
  model <- loadModel

  model$Score <- ceiling(model$RiskScore*10)
  print(model$Pgh_FireDistrict)
  
  # get data subset
  data <- reactive({
    # default option: select all
   
    print("Filtering Fire Risk Scores")

    d <- subset(model, subset = (Score <= input$range[2] & Score >= input$range[1]))

    # filter by property type (STATEDESC)
    if (!("All Classification Types" %in% input$property)){
      d <- subset(d, subset=(state_desc %in% input$property))
    }
    
    # filter by usage type (USEDESC)
    if (!("All Usage Types" %in% input$use)) {
      d <- subset(d, subset=(use_desc %in% input$use))
    }
    
    # filter by neighborhood (NEIGHDESC)
    if (!("All Neighborhoods" %in% input$nbhood)) {
      d <- subset(d, subset=(geo_name_nhood %in% input$nbhood))
    }
    # filter by fire district
    if (!("All Fire Districts" %in% input$fire_dst)){
      d <- subset(d, subset=(Pgh_FireDistrict %in% input$fire_dst)) 
    }
    d
  
     
    
  })
  
  #### Visualization output  ####
  plot_Vis <- reactive({
    
      if (input$xvar == "Property Classification") {
        x_axis <- "state_desc"
      } else if (input$xvar == "Property Usage Type") {
        x_axis <- "use_desc"
      } else if (input$xvar == "Neighborhood") {
        x_axis <- "geo_name_nhood"
      }
      
      if (input$yvar == "Fire Risk Scores") {
        if (input$xvar == "Fire District") {
        x_axis <- "Pgh_FireDistrict"
      } 
     
    } 
    
    y_axis <- input$yvar

    
    
    
    ## Create visualization ##
    
    if (y_axis == "Fire Risk Scores"){
      print("displaying Fire risk")
      
     
      # consider average risk score by x axis
      if (nlevels(data()[[x_axis]]) <= 15){
        plot <- ggplot(data = data()[!is.na(data()[[x_axis]]),],aes(x=data()[!is.na(data()[[x_axis]]),][[x_axis]],y=Score)) + 
          stat_summary(fun.y = "mean",geom = "bar",width=0.8,fill="steelblue") + 
          stat_summary(aes(label=..y..),fun.y = function(x){round(mean(x),2)},geom = "text",size=5,vjust=-1,width=0.8) +
          theme(plot.title = element_text(size = 18, face = "bold"),text = element_text(size=15)) +
          ggtitle("Average Risk Score") + ylim(0,10) + 
          xlab(x_axis) +
          ylab("Risk Score")
        
      }else{
        data_selected <- data()[!is.na(data()[[x_axis]]),]
        
        ag_score <- aggregate(data_selected[["Score"]] ~ data_selected[[x_axis]], data_selected, mean)
        ag_label <- as.vector(unlist(ag_score[order(ag_score[2]),][1]))
        print(length(ag_label))
        # h = 550 + 10 * length(ag_label)
        plot <- ggplot(data = data_selected, aes(x=data_selected[[x_axis]],y=Score)) + 
          stat_summary(fun.y = "mean",geom = "bar",width=0.8,fill="steelblue") + 
          coord_flip() + scale_x_discrete(limits=ag_label,labels=ag_label) +
          stat_summary(aes(label=..y..),fun.y = function(x){round(mean(x),2)},geom = "text",size=4,hjust=-1) +
          ggtitle("Average Risk Score") + ylim(0,10) + 
          theme(plot.title = element_text(size = 18, face = "bold"),text = element_text(size=14)) +
          xlab(x_axis) +
          ylab("Risk Score")
      }
      
    }
    plot
  })

  

    
  ## Allow for variable height of visualization, dependent on number of records ##
  
  myHeight <- function(){
    
    if (input$yvar == "Fire Risk Scores") {
      
        if (input$xvar == "Property Classification") {
          x_axis <- "state_desc"
        } else if (input$xvar == "Property Usage Type") {
          x_axis <- "use_desc"
        } else if (input$xvar == "Neighborhood") {
          x_axis <- "geo_name_nhood"
        } else if (input$xvar == "Fire District") {
          x_axis <- "Pgh_FireDistrict"
        } 
        data_selected <- data()[!is.na(data()[[x_axis]]),]
      
        ag_score <- aggregate(data_selected[["Score"]] ~ data_selected[[x_axis]], data_selected, mean)
        ag_label <- as.vector(unlist(ag_score[order(ag_score[2]),][1]))
    
    } 
    
    num = length(ag_label)
    print(num)
    
    if (num < 20) {
      modifier = 3
    } else {
      modifier = 200
    }
    print(modifier)
    return(650 + 5 * modifier)
  }
  
  # visualization plot
  output$distPlot <- renderPlot({
    plot_Vis()
  }, 
  
  width=850,height=myHeight)

  
  
  # download plotVis
  output$plotVis <- downloadHandler(
    
    filename <- "visualization.png",
    content <- function(file){
      dpi_val <- 85
      h <- (myHeight()/dpi_val)
      print(myHeight())
      print(h)
      ggsave(file, plot = plot_Vis(),device = "png",width=11, height=h,units="in",dpi=dpi_val)
    }
  )
  
 
  # download table
  output$downloadTable <- downloadHandler(
    filename = "table.csv",
    content = function(file) {
      write.csv(as.data.frame(data())[c(1:3,86,13,15,17,23,25,99,129)], file)
    }
  )
  
  # print total number of records selected (for reference)
  output$n_records <- renderText({
    nrow(data())
  })

  

})
