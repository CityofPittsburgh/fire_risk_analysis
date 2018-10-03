FROM ipds/shiny-base

# copy the app to the image
RUN mkdir /root/firerisk-dashboard
COPY firerisk-dashboard /root/firerisk-dashboard

CMD ["R", "-e shiny::runApp('/root/firerisk-dashboard')"]