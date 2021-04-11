# README

This README would normally document whatever steps are necessary to get the
application up and running.

Things you may want to cover:

* System version
  - Ruby: 2.5.1
  - Rails: 5.2.5
  - Postgres: 12.6
  - Elasticsearch: 7.11.2

* Setup
  1) Ensure you have Ruby, Rails, Postgres, Elasticsearch installed
  2) Run `bundle install` to install all ruby gems
  3) Set up database (Refer to "Database initialization")

* Database initialization
  1) Edit the `username` and `password` fields in "database.yml" file
     ```yml
     development:
      <<: *default
      database: cz4034_development
      host: localhost
      username: YOUR_POSTGRES_USERNAME
      password: YOUR_POSTGRES_PASSWORD
     ```
  2) Run `rails db:reset`
  3) Run

* How to run the server
  1) Run `rails s`

* API endpoints
  - `/search?search_term=randomstuff&hashtags[]=hashtag1&hashtags[]=hashtag2`
